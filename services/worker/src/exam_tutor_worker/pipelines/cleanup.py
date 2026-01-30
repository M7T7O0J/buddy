from __future__ import annotations

import re
from collections import Counter


def cleanup_markdown(md: str) -> str:
    """Lightweight pre-chunk cleanup to reduce garbage + duplicate tokens.

    This intentionally stays conservative (rules-first) to avoid deleting real content.
    It focuses on removing *repeated* short lines which are commonly headers/footers.
    """
    md = md.replace("\r\n", "\n")
    lines = md.splitlines()

    def is_candidate(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith("#"):
            return False
        if s.startswith("|"):  # tables
            return False
        if re.match(r"^(\s*[-*+]\s+|\s*\d+\.\s+)", s):
            return False
        if s == "[IMAGE]":
            return False
        # keep short-ish lines (header/footer-like)
        return 3 <= len(s) <= 80

    counts = Counter([ln.strip() for ln in lines if is_candidate(ln)])
    repeated = {s for s, c in counts.items() if c >= 6}

    if not repeated:
        return md.strip()

    out = []
    for ln in lines:
        s = ln.strip()
        if s in repeated:
            continue
        out.append(ln)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

