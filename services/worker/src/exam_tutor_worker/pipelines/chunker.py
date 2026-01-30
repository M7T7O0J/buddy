from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _Block:
    text: str
    token_count: int
    section_path: Optional[str]
    parent_section_path: Optional[str]
    block_type: str  # text | table | code | list | image
    heading_level: Optional[int] = None


class TokenCounter:
    """Token counter abstraction with a safe fallback.

    If we can load a Hugging Face tokenizer, we use it.
    Otherwise, we fall back to a robust heuristic (≈4 chars/token).
    """

    def __init__(self, tokenizer_name: str):
        self._tok = None
        if AutoTokenizer is not None:
            try:
                self._tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            except Exception:
                self._tok = None

    def count(self, text: str) -> int:
        if self._tok is not None:
            return int(len(self._tok.encode(text, add_special_tokens=False)))
        # heuristic fallback
        text = text.strip()
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def tail_text(self, text: str, tail_tokens: int) -> str:
        if self._tok is None:
            # fallback: take last ~tail_tokens*4 chars
            n = max(0, tail_tokens * 4)
            return text[-n:]
        ids = self._tok.encode(text, add_special_tokens=False)
        tail_ids = ids[-tail_tokens:]
        return self._tok.decode(tail_ids)


class TokenAwareChunker:
    """Hierarchical, structure-aware, token-aware chunker for Docling Markdown.

    Strategy:
    - parse markdown into blocks (heading / text / list / table / code / image)
    - group by "parent section" (heading level <= parent_section_level) so chunks never cross major section boundaries
    - within each parent section, accumulate blocks until max_tokens
    - overlap by overlap_tokens using token tail (within the same parent section)
    - recursively split oversized text blocks (paragraphs -> sentences -> hard split)
    """

    def __init__(
        self,
        tokenizer_name: str,
        min_tokens: int,
        max_tokens: int,
        overlap_tokens: int,
        *,
        parent_section_level: int = 2,
    ):
        self.tc = TokenCounter(tokenizer_name)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.parent_section_level = max(1, min(6, int(parent_section_level)))

    def _tok_count(self, text: str) -> int:
        return self.tc.count(text)

    def _section_path(self, stack: List[Tuple[int, str]]) -> Optional[str]:
        if not stack:
            return None
        return " > ".join(title for _, title in stack)

    def _parent_section_path(self, stack: List[Tuple[int, str]]) -> Optional[str]:
        if not stack:
            return None
        kept = [(lvl, title) for (lvl, title) in stack if lvl <= self.parent_section_level]
        if not kept:
            # If the doc starts with a deeper heading (rare), fall back to the top-most heading we have.
            kept = [stack[0]]
        return " > ".join(title for _, title in kept)

    def _iter_blocks(self, text: str) -> List[_Block]:
        lines = text.splitlines()
        blocks: List[_Block] = []
        para: List[str] = []
        section_stack: List[Tuple[int, str]] = []

        def flush_para() -> None:
            nonlocal para
            if not para:
                return
            s = "\n".join(para).strip()
            para = []
            if not s:
                return
            blocks.append(
                _Block(
                    text=s,
                    token_count=self._tok_count(s),
                    section_path=self._section_path(section_stack),
                    parent_section_path=self._parent_section_path(section_stack),
                    block_type="text",
                )
            )

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Blank line flushes paragraph.
            if not stripped:
                flush_para()
                i += 1
                continue

            # Heading updates section stack.
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if m:
                flush_para()
                level = len(m.group(1))
                title = m.group(2).strip()
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, title))
                heading_text = line.strip()
                blocks.append(
                    _Block(
                        text=heading_text,
                        token_count=self._tok_count(heading_text),
                        section_path=self._section_path(section_stack),
                        parent_section_path=self._parent_section_path(section_stack),
                        block_type="heading",
                        heading_level=level,
                    )
                )
                i += 1
                continue

            # Image marker (normalized in normalize_markdown).
            if stripped == "[IMAGE]":
                flush_para()
                blocks.append(
                    _Block(
                        text="[IMAGE]",
                        token_count=1,
                        section_path=self._section_path(section_stack),
                        parent_section_path=self._parent_section_path(section_stack),
                        block_type="image",
                    )
                )
                i += 1
                continue

            # Code fence block.
            if stripped.startswith("```") or stripped.startswith("~~~"):
                flush_para()
                fence = stripped[:3]
                code_lines = [line]
                i += 1
                while i < len(lines):
                    code_lines.append(lines[i])
                    if lines[i].strip().startswith(fence):
                        i += 1
                        break
                    i += 1
                code = "\n".join(code_lines).strip()
                blocks.append(
                    _Block(
                        text=code,
                        token_count=self._tok_count(code),
                        section_path=self._section_path(section_stack),
                        parent_section_path=self._parent_section_path(section_stack),
                        block_type="code",
                    )
                )
                continue

            # Markdown table block (simple heuristic).
            if stripped.startswith("|") and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("|") and re.search(r"-{3,}", next_line):
                    flush_para()
                    table_lines = [line, lines[i + 1]]
                    i += 2
                    while i < len(lines) and lines[i].strip().startswith("|"):
                        table_lines.append(lines[i])
                        i += 1
                    table = "\n".join(table_lines).strip()
                    blocks.append(
                        _Block(
                            text=table,
                            token_count=self._tok_count(table),
                            section_path=self._section_path(section_stack),
                            parent_section_path=self._parent_section_path(section_stack),
                            block_type="table",
                        )
                    )
                    continue

            # List block.
            if re.match(r"^(\s*[-*+]\s+|\s*\d+\.\s+)", line):
                flush_para()
                list_lines = [line]
                i += 1
                while i < len(lines):
                    nxt = lines[i]
                    if not nxt.strip():
                        break
                    # Continuation or nested list
                    if re.match(r"^(\s+|\s*[-*+]\s+|\s*\d+\.\s+)", nxt):
                        list_lines.append(nxt)
                        i += 1
                        continue
                    break
                lst = "\n".join(list_lines).strip()
                blocks.append(
                    _Block(
                        text=lst,
                        token_count=self._tok_count(lst),
                        section_path=self._section_path(section_stack),
                        parent_section_path=self._parent_section_path(section_stack),
                        block_type="list",
                    )
                )
                continue

            # Default: accumulate into paragraph.
            para.append(line)
            i += 1

        flush_para()
        return blocks

    def _split_big_text(self, block: str) -> List[str]:
        # First split by paragraphs; then by sentences; finally by hard length.
        parts = [p.strip() for p in re.split(r"\n{2,}", block.strip()) if p.strip()]
        out: List[str] = []
        for p in parts:
            if self._tok_count(p) <= self.max_tokens:
                out.append(p)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", p)
            cur = ""
            for s in sentences:
                if not s:
                    continue
                cand = (cur + " " + s).strip() if cur else s
                if self._tok_count(cand) <= self.max_tokens:
                    cur = cand
                    continue
                if cur:
                    out.append(cur)
                    cur = ""
                # Hard split long "sentence" if needed.
                if self._tok_count(s) <= self.max_tokens:
                    cur = s
                    continue
                # fallback: chunk by approx chars
                step = max(1, self.max_tokens * 4)
                for j in range(0, len(s), step):
                    piece = s[j : j + step].strip()
                    if piece:
                        out.append(piece)
            if cur:
                out.append(cur)
        return out

    def chunk(self, text: str) -> List[Chunk]:
        blocks = self._iter_blocks(text)

        # Expand oversized blocks (only for text; keep tables/code intact where possible).
        expanded: List[_Block] = []
        for b in blocks:
            if b.token_count <= self.max_tokens:
                expanded.append(b)
                continue
            if b.block_type in {"table", "code"}:
                # Fall back to hard splits if a single table/code block is enormous.
                parts = self._split_big_text(b.text)
                for p in parts:
                    expanded.append(
                        _Block(
                            text=p,
                            token_count=self._tok_count(p),
                            section_path=b.section_path,
                            parent_section_path=b.parent_section_path,
                            block_type=b.block_type,
                        )
                    )
                continue
            parts = self._split_big_text(b.text)
            for p in parts:
                expanded.append(
                    _Block(
                        text=p,
                        token_count=self._tok_count(p),
                        section_path=b.section_path,
                        parent_section_path=b.parent_section_path,
                        block_type=b.block_type,
                    )
                )
        blocks = expanded

        chunks: List[Chunk] = []
        cur: List[_Block] = []
        cur_tokens = 0
        idx = 0
        active_parent: Optional[str] = None

        def flush() -> None:
            nonlocal idx, cur, cur_tokens
            if not cur:
                return
            joined = "\n\n".join(b.text for b in cur).strip()
            t = self._tok_count(joined)
            if t >= 1:
                section_paths: List[str] = []
                for b in cur:
                    if b.section_path and (not section_paths or section_paths[-1] != b.section_path):
                        section_paths.append(b.section_path)
                block_types: List[str] = []
                for b in cur:
                    if not block_types or block_types[-1] != b.block_type:
                        block_types.append(b.block_type)
                chunks.append(
                    Chunk(
                        index=idx,
                        text=joined,
                        token_count=t,
                        metadata={
                            "section_paths": section_paths,
                            "primary_section_path": section_paths[0] if section_paths else None,
                            "parent_section_path": active_parent,
                            "block_types": block_types,
                        },
                    )
                )
                idx += 1
            cur = []
            cur_tokens = 0

        for b in blocks:
            # Parent-section boundary: if a heading <= parent_section_level is encountered, we start a new group.
            if b.block_type == "heading" and b.heading_level is not None and b.heading_level <= self.parent_section_level:
                flush()
                active_parent = b.parent_section_path

            # If we have no active parent yet (document preface before first heading), set it from the first block.
            if active_parent is None:
                active_parent = b.parent_section_path

            bt = b.token_count
            if cur_tokens + bt <= self.max_tokens:
                cur.append(b)
                cur_tokens += bt
            else:
                flush()

                # Only overlap within the same parent section.
                if chunks and self.overlap_tokens > 0 and chunks[-1].metadata.get("parent_section_path") == active_parent:
                    tail = self.tc.tail_text(chunks[-1].text, self.overlap_tokens).strip()
                    if tail:
                        cur = [
                            _Block(
                                text=tail,
                                token_count=self._tok_count(tail),
                                section_path=None,
                                parent_section_path=active_parent,
                                block_type="text",
                            ),
                            b,
                        ]
                    else:
                        cur = [b]
                    cur_tokens = self._tok_count("\n\n".join(x.text for x in cur))
                else:
                    cur = [b]
                    cur_tokens = bt

        flush()

        # merge small tail chunks
        merged: List[Chunk] = []
        for c in chunks:
            if (
                merged
                and c.token_count < self.min_tokens
                and (merged[-1].metadata.get("parent_section_path") == c.metadata.get("parent_section_path"))
            ):
                prev = merged.pop()
                new_text = prev.text + "\n\n" + c.text
                new_tokens = self._tok_count(new_text)
                merged.append(
                    Chunk(
                        index=prev.index,
                        text=new_text,
                        token_count=new_tokens,
                        metadata={
                            "section_paths": list(
                                dict.fromkeys(
                                    [*(prev.metadata.get("section_paths") or []), *(c.metadata.get("section_paths") or [])]
                                )
                            ),
                            "primary_section_path": (prev.metadata.get("primary_section_path") or c.metadata.get("primary_section_path")),
                            "parent_section_path": (prev.metadata.get("parent_section_path") or c.metadata.get("parent_section_path")),
                            "block_types": list(
                                dict.fromkeys([*(prev.metadata.get("block_types") or []), *(c.metadata.get("block_types") or [])])
                            ),
                        },
                    )
                )
            else:
                merged.append(c)

        # reindex
        return [Chunk(index=i, text=c.text, token_count=c.token_count, metadata=c.metadata) for i, c in enumerate(merged)]
