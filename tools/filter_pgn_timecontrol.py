#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

TC_RE = re.compile(r'^\[TimeControl\s+"([^"]+)"\]\s*$')

def iter_pgn_games(path: Path) -> Iterator[str]:
    """Stream PGN games as raw text blocks."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        pending = None
        while True:
            line = pending if pending is not None else f.readline()
            pending = None
            if not line:
                break

            while line and not line.startswith("["):
                line = f.readline()
            if not line:
                break

            game_lines: List[str] = []
            prev_blank = False
            while True:
                if not line:
                    break
                if prev_blank and line.startswith("[") and game_lines:
                    pending = line
                    break
                game_lines.append(line)
                prev_blank = (line.strip() == "")
                line = f.readline()

            game = "".join(game_lines).strip()
            if game:
                yield game + "\n\n"

def parse_timecontrol(tc: str) -> Tuple[Optional[int], Optional[int]]:
    """
    chess.com often uses:
      "180" or "60+1" or "1500+10"
    Return (base_seconds, increment_seconds) or (None,None) if unknown/daily/etc.
    """
    tc = (tc or "").strip()
    if not tc:
        return None, None
    # daily formats often contain "/", skip them unless you want to handle separately
    if "/" in tc:
        return None, None
    if "+" in tc:
        a, b = tc.split("+", 1)
        a, b = a.strip(), b.strip()
        if a.isdigit() and b.isdigit():
            return int(a), int(b)
        return None, None
    if tc.isdigit():
        return int(tc), 0
    return None, None

def extract_tc(game: str) -> str:
    for line in game.splitlines():
        m = TC_RE.match(line.strip())
        if m:
            return m.group(1)
    return ""

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-base", type=int, default=0, help="Keep only games with base >= this (seconds).")
    ap.add_argument("--max-base", type=int, default=0, help="Keep only games with base <= this (seconds). 0 disables.")
    ap.add_argument("--min-inc", type=int, default=0, help="Keep only games with increment >= this (seconds).")
    ap.add_argument("--keep-unknown", action="store_true", help="Keep games where TimeControl is missing/unknown.")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = kept = unknown = 0

    with out.open("w", encoding="utf-8", newline="") as w:
        for game in iter_pgn_games(inp):
            total += 1
            tc = extract_tc(game)
            base, inc = parse_timecontrol(tc)

            if base is None:
                unknown += 1
                if args.keep_unknown:
                    w.write(game)
                    kept += 1
                continue

            if args.min_base and base < args.min_base:
                continue
            if args.max_base and base > args.max_base:
                continue
            if args.min_inc and (inc or 0) < args.min_inc:
                continue

            w.write(game)
            kept += 1

    print("Done.")
    print(f"Total games scanned: {total:,}")
    print(f"Games kept:         {kept:,}")
    print(f"Unknown TimeControl:{unknown:,}")
    print(f"Output: {out}")

if __name__ == "__main__":
    main()
