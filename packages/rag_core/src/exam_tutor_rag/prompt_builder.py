from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

from exam_tutor_common.constants import TutorMode

from .safety_policy import GROUNDED_TUTOR_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptParts:
    system: str
    user: str


def _format_sources(sources: List[dict]) -> str:
    lines: List[str] = []
    for s in sources:
        meta = s.get("metadata") or {}
        section = meta.get("parent_section_path") or meta.get("primary_section_path")
        section_part = f", section={section}" if section else ""
        header = (
            f"[chunk:{s['chunk_id']}] {s['source_title']} "
            f"(exam={s['exam']}, subject={s.get('subject')}, topic={s.get('topic')}{section_part})"
        )
        lines.append(header)
        lines.append(s['content'])
        lines.append("---")
    return "\n".join(lines).strip()


def build_prompt(
    *,
    mode: TutorMode,
    exam: str,
    language: str,
    question: str,
    sources: List[dict],
    memory: Optional[str] = None,
) -> PromptParts:
    """Build a single-turn prompt with grounding + (optional) memory.

    We keep the structure simple for maximum compatibility with OpenAI-compatible servers.
    """
    mode_hint = {
        TutorMode.doubt: "Answer as a teacher. Show steps and explain why each step is taken.",
        TutorMode.practice: "Create practice: give a question, then hints, then a full solution. Keep it exam-style.",
        TutorMode.pyq: "Answer like PYQ trainer: show approach, key formula/framework, and a final solution.",
    }[mode]

    sources_block = _format_sources(sources) if sources else "(no sources retrieved)"

    memory_block = f"\n\nCHAT MEMORY (summary + recent context):\n{memory}" if memory else ""

    system = (
        GROUNDED_TUTOR_SYSTEM_PROMPT
        + f"\n\nMode instructions: {mode_hint}"
        + f"\nLanguage: {language}"
        + f"\nTarget exam: {exam}"
    )

    user = (
        "SOURCES:\n"
        + sources_block
        + memory_block
        + "\n\nUSER QUESTION:\n"
        + question
        + "\n\nRESPONSE REQUIREMENTS:\n"
        + "- If you use facts/formulas/articles, cite chunk ids like [chunk:123].\n"
        + "- If sources are insufficient, say so and ask a clarifying question.\n"
    )

    return PromptParts(system=system, user=user)


def to_openai_messages(parts: PromptParts) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": parts.system},
        {"role": "user", "content": parts.user},
    ]
