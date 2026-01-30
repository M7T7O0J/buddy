from __future__ import annotations

from exam_tutor_common.constants import TutorMode


def style_hint(mode: TutorMode, exam: str) -> str:
    """A small extra hint for style depending on exam."""
    if exam.startswith("UPSC"):
        return (
            "Write in UPSC style: intro (1-2 lines), body (headings/bullets), conclusion. "
            "Use examples and constitutional references when relevant."
        )
    # Default to GATE-like
    return (
        "Write in GATE style: define concept, list given/required, show steps with formulas, then final answer. "
        "Include common mistakes and quick checks when helpful."
    )
