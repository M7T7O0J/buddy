from __future__ import annotations

import re


def normalize_markdown(md: str) -> str:
    """Normalize Markdown extracted from Docling.

    MVP normalization:
    - remove excessive blank lines
    - strip trailing whitespace
    - keep headings/lists (useful for UPSC structure)
    """
    md = md.replace("\r\n", "\n")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)

    # Normalize Docling image placeholders to a consistent, chunker-friendly marker.
    md = md.replace("<!-- image -->", "[IMAGE]")

    # Remove consecutive duplicate paragraphs (common in OCR/PDF exports).
    paras = [p.strip() for p in re.split(r"\n{2,}", md) if p.strip()]
    deduped = []
    for p in paras:
        if deduped and deduped[-1] == p:
            continue
        deduped.append(p)

    md = "\n\n".join(deduped)
    return md.strip()
