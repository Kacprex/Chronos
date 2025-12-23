from __future__ import annotations

import os


def read_generation(path: str) -> int:
    """Read generation integer from a text file.

    If the file does not exist or is invalid, returns 0.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return int(s)
    except Exception:
        return 0


def write_generation(path: str, gen: int) -> None:
    """Write generation integer to a text file (atomic-ish)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(gen)))
    os.replace(tmp, path)

