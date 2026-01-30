from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .chunker import Chunk


_FRONT_MATTER_KEYWORDS = [
    "acknowledgements",
    "foreword",
    "publication",
    "publication team",
    "offices of the publication division",
    "all rights reserved",
    "isbn",
    "pd ",
    "first edition",
    "reprinted",
    "revised edition",
    "textbook in",
    "national council of educational research",
    "textbook development committee",
    "chairperson",
    "chief advisor",
    "advisors",
    "team members",
    "member-coordinators",
]

_BOILERPLATE_PATTERNS = [
    r"\ball rights reserved\b",
    r"\bno part of this publication may be reproduced\b",
    r"\bprinted on\b",
    r"\bpublished at\b",
    r"\bphone\s*:\b",
]


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = True
    min_tokens: int = 40
    max_chunks_per_doc: int = 2000
    max_chunks_per_parent: int = 400
    drop_front_matter: bool = True
    drop_image_only: bool = True
    drop_boilerplate: bool = True


@dataclass(frozen=True)
class FilterStats:
    total_in: int
    total_out: int
    dropped: int
    dropped_by_tag: Dict[str, int]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_like_heading_only(text: str) -> bool:
    t = text.strip()
    if not t.startswith("#"):
        return False
    # Heading only (no body)
    return "\n" not in t.strip()


def _is_image_only(text: str) -> bool:
    t = text.strip()
    return t == "[IMAGE]"


def _section_tag(parent_section_path: Optional[str]) -> str:
    return (parent_section_path or "").strip().lower()


def _match_any_keyword(hay: str, keywords: Iterable[str]) -> bool:
    h = hay.lower()
    return any(k in h for k in keywords)


def _has_boilerplate(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _BOILERPLATE_PATTERNS)


def _compute_quality_score(*, token_count: int, tags: List[str]) -> float:
    score = 1.0
    if "front_matter" in tags:
        score -= 0.7
    if "boilerplate" in tags:
        score -= 0.6
    if "image_only" in tags:
        score -= 0.7
    if "duplicate" in tags:
        score -= 1.0
    if "low_signal" in tags:
        score -= 0.3
    if token_count >= 200:
        score += 0.05
    return max(0.0, min(1.0, score))


def _merge_heading_with_next(chunks: List[Chunk]) -> List[Chunk]:
    out: List[Chunk] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if i + 1 < len(chunks) and _looks_like_heading_only(c.text):
            n = chunks[i + 1]
            # Only merge if same parent section.
            if (c.metadata or {}).get("parent_section_path") == (n.metadata or {}).get("parent_section_path"):
                merged_text = c.text.strip() + "\n\n" + n.text.strip()
                merged_tokens = max(int(len(merged_text) / 4), c.token_count + n.token_count)
                merged_meta = dict(n.metadata or {})
                merged_meta.setdefault("merged_heading", True)
                out.append(Chunk(index=c.index, text=merged_text, token_count=merged_tokens, metadata=merged_meta))
                i += 2
                continue
        out.append(c)
        i += 1
    # reindex to keep chunk_index dense
    return [Chunk(index=i, text=c.text, token_count=c.token_count, metadata=c.metadata) for i, c in enumerate(out)]


def filter_chunks(
    chunks: List[Chunk],
    *,
    cfg: FilterConfig,
) -> Tuple[List[Chunk], FilterStats]:
    if not cfg.enabled:
        return chunks, FilterStats(total_in=len(chunks), total_out=len(chunks), dropped=0, dropped_by_tag={})

    chunks = _merge_heading_with_next(chunks)

    dropped_by_tag: Dict[str, int] = {}
    seen_hashes: set[str] = set()
    kept: List[Chunk] = []

    for c in chunks:
        tags: List[str] = []
        meta = dict(c.metadata or {})
        parent = meta.get("parent_section_path") or meta.get("primary_section_path") or ""
        section = _section_tag(parent)

        if cfg.drop_front_matter and _match_any_keyword(section, _FRONT_MATTER_KEYWORDS):
            tags.append("front_matter")

        if cfg.drop_boilerplate and _has_boilerplate(c.text):
            tags.append("boilerplate")

        if cfg.drop_image_only and _is_image_only(c.text):
            tags.append("image_only")

        if c.token_count < cfg.min_tokens:
            tags.append("low_signal")

        h = _sha(c.text.strip())
        if h in seen_hashes:
            tags.append("duplicate")

        quality_score = _compute_quality_score(token_count=c.token_count, tags=tags)
        meta["tags"] = sorted(set([*(meta.get("tags") or []), *tags]))
        meta["quality_score"] = quality_score

        drop = False
        if "duplicate" in tags:
            drop = True
        if "front_matter" in tags:
            drop = True
        if "boilerplate" in tags:
            drop = True
        if "image_only" in tags:
            drop = True

        if drop:
            for t in set(tags):
                dropped_by_tag[t] = dropped_by_tag.get(t, 0) + 1
            continue

        seen_hashes.add(h)
        kept.append(Chunk(index=c.index, text=c.text, token_count=c.token_count, metadata=meta))

    # Cap per parent section (keep highest-quality chunks within each parent section).
    by_parent: Dict[str, List[Chunk]] = {}
    for c in kept:
        parent = (c.metadata or {}).get("parent_section_path") or ""
        by_parent.setdefault(parent, []).append(c)

    capped: List[Chunk] = []
    for parent, group in by_parent.items():
        if len(group) <= cfg.max_chunks_per_parent:
            capped.extend(group)
            continue
        group_sorted = sorted(group, key=lambda x: ((x.metadata or {}).get("quality_score", 0.0), x.token_count), reverse=True)
        keep_set = set(_sha(c.text.strip()) for c in group_sorted[: cfg.max_chunks_per_parent])
        for c in group:
            if _sha(c.text.strip()) in keep_set:
                capped.append(c)
            else:
                dropped_by_tag["cap_parent"] = dropped_by_tag.get("cap_parent", 0) + 1

    # Cap overall doc.
    if len(capped) > cfg.max_chunks_per_doc:
        capped_sorted = sorted(capped, key=lambda x: ((x.metadata or {}).get("quality_score", 0.0), x.token_count), reverse=True)
        capped = capped_sorted[: cfg.max_chunks_per_doc]
        dropped_by_tag["cap_doc"] = dropped_by_tag.get("cap_doc", 0) + (len(capped_sorted) - len(capped))

    # Reindex after filtering/capping.
    capped = [Chunk(index=i, text=c.text, token_count=c.token_count, metadata=c.metadata) for i, c in enumerate(capped)]

    stats = FilterStats(
        total_in=len(chunks),
        total_out=len(capped),
        dropped=len(chunks) - len(capped),
        dropped_by_tag=dropped_by_tag,
    )
    return capped, stats
