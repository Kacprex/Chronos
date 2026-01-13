from __future__ import annotations
import os
from pathlib import Path

def data_root() -> Path:
    # Default to E:\chronos if not specified
    v = os.environ.get("CHRONOS_DATA", r"E:\chronos")
    return Path(v)

def ensure_layout() -> dict[str, Path]:
    root = data_root()
    private = root / "private"
    public = root / "public"

    paths = {
        "root": root,
        "private": private,
        "public": public,
        "sl": private / "sl",
        "models": private / "models",
        "logs": private / "logs",
        "control": private / "control",
        "bin": private / "bin",
        "public_metrics": public / "metrics",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # seed a sample_fens file if missing
    sample_fens = public / "sample_fens.txt"
    if not sample_fens.exists():
        sample_fens.write_text(
            "\n".join([
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 2 8",
            ]) + "\n",
            encoding="utf-8",
        )

    return paths
