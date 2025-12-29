#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import os
import random
from pathlib import Path
from typing import Iterator, List, Tuple, TextIO


def open_text(path: Path) -> TextIO:
    """Open .pgn or .pgn.gz as text."""
    if path.name.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def iter_pgn_games(path: Path) -> Iterator[str]:
    """
    Stream PGN games as raw text blocks.
    Splits on typical PGN boundaries: a new header line '[' after a blank line.
    """
    with open_text(path) as f:
        pending = None
        while True:
            line = pending if pending is not None else f.readline()
            pending = None
            if not line:
                break

            # Skip until a header tag line
            while line and not line.startswith("["):
                line = f.readline()

            if not line:
                break

            game_lines: List[str] = []
            prev_blank = False

            # Read until next game's header (after a blank line), or EOF
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


def reservoir_sample_games(path: Path, k: int, rng: random.Random) -> Tuple[List[str], int]:
    """
    Reservoir-sample up to k games from a PGN file in one pass.
    Returns (sampled_games, total_games_seen).
    """
    sample: List[str] = []
    n = 0
    for game in iter_pgn_games(path):
        n += 1
        if k <= 0:
            continue
        if n <= k:
            sample.append(game)
        else:
            j = rng.randint(1, n)
            if j <= k:
                sample[j - 1] = game
    return sample, n


def default_players_dir() -> Path:
    # Prefer CHRONOS_DATA_ROOT if set, otherwise default E:\chronos
    root = os.environ.get("CHRONOS_DATA_ROOT", r"E:\chronos")
    return Path(root) / "datasets" / "pgn" / "players"


def default_output_file() -> Path:
    root = os.environ.get("CHRONOS_DATA_ROOT", r"E:\chronos")
    return Path(root) / "datasets" / "pgn" / "elite_mix.pgn"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create elite_mix.pgn by sampling games from E:\\chronos\\datasets\\pgn\\players\\*.pgn"
    )
    ap.add_argument(
        "--input-dir",
        default=str(default_players_dir()),
        help="Directory containing per-player PGNs (default uses CHRONOS_DATA_ROOT or E:\\chronos).",
    )
    ap.add_argument(
        "--pattern",
        default="*.pgn",
        help="Glob pattern in input-dir (default: *.pgn). Use *.pgn or *.pgn* if you have .gz too.",
    )
    ap.add_argument(
        "--output",
        default=str(default_output_file()),
        help="Output PGN file (default: ...\\elite_mix.pgn).",
    )
    ap.add_argument(
        "--per-file",
        type=int,
        default=10000,
        help="Max games sampled per input file (default: 10000).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    ap.add_argument(
        "--shuffle-within-file",
        action="store_true",
        help="Shuffle sampled games inside each file before writing.",
    )
    ap.add_argument(
        "--shuffle-file-order",
        action="store_true",
        help="Shuffle the order of input files before processing/writing.",
    )
    ap.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="Skip input files with fewer than this many total games (default: 1).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan & report counts but do not write output.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found: {input_dir}\\{args.pattern}")

    rng = random.Random(args.seed)
    if args.shuffle_file_order:
        rng.shuffle(files)

    # Ensure output folder exists
    output.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    report: List[Tuple[str, int, int]] = []  # (file, total_seen, sampled)

    # Collect + write
    if args.dry_run:
        for fp in files:
            sample, total_seen = reservoir_sample_games(fp, min(args.per_file, 1), rng)
            # ^ tiny sample just to count quickly? No: counting requires full scan anyway.
            # But we still want full scan counts. We'll do real scan with k=0 to count.
        # We'll just do the same pass below with writing disabled.

    out_fh = None if args.dry_run else output.open("w", encoding="utf-8", newline="")

    try:
        for idx, fp in enumerate(files, start=1):
            sample, total_seen = reservoir_sample_games(fp, args.per_file, rng)

            if total_seen < args.min_games:
                report.append((fp.name, total_seen, 0))
                continue

            if args.shuffle_within_file and len(sample) > 1:
                rng.shuffle(sample)

            report.append((fp.name, total_seen, len(sample)))

            if not args.dry_run and out_fh is not None and sample:
                for g in sample:
                    out_fh.write(g)
                total_written += len(sample)

            if idx % 10 == 0 or idx == len(files):
                print(f"[progress] files={idx}/{len(files)}  games_written={total_written:,}")

    finally:
        if out_fh is not None:
            out_fh.flush()
            out_fh.close()

    print("\nDone.")
    print(f"Input dir:  {input_dir}")
    print(f"Pattern:    {args.pattern}")
    print(f"Per-file:   {args.per_file}")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {output}")
    print("\nPer-file summary (total_seen -> sampled):")
    for name, total_seen, sampled in sorted(report, key=lambda x: (-x[2], -x[1], x[0].lower())):
        print(f"  {name}: {total_seen:,} -> {sampled:,}")

    if not args.dry_run:
        print(f"\nTotal games written: {total_written:,}")


if __name__ == "__main__":
    main()
