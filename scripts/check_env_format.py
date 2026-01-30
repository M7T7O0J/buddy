from __future__ import annotations

import re
import sys
from pathlib import Path


VAR_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ".env")
    if not path.exists():
        print(f"env file not found: {path}", file=sys.stderr)
        return 2

    bad: list[tuple[int, str]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not VAR_RE.match(line):
            preview = stripped
            if len(preview) > 120:
                preview = preview[:120] + "…"
            bad.append((i, preview))

    if bad:
        print(f"Invalid lines in {path} (must be KEY=VALUE or start with #):", file=sys.stderr)
        for (line_no, preview) in bad:
            print(f"- L{line_no}: {preview}", file=sys.stderr)
        return 1

    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

