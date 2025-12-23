"""Analyze RL buffer shards for diversity/quality.

PowerShell examples:
  python tools/analyze_rl_buffer.py
  python tools/analyze_rl_buffer.py --out debug/rl_buffer_report.csv
  python tools/analyze_rl_buffer.py --min-unique 0.2
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import torch

from src.config import RL_BUFFER_DIR
from src.debug.shard_stats import analyze_rl_shard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default=RL_BUFFER_DIR, help="RL buffer directory")
    ap.add_argument("--out", default="", help="Optional CSV output path")
    ap.add_argument("--min-unique", type=float, default=None, help="Highlight shards below this unique ratio")
    args = ap.parse_args()

    buf = Path(args.buffer)
    if not buf.exists():
        print(f"Buffer dir not found: {buf}")
        return 2

    shard_files: List[Path] = sorted(buf.glob("rl_shard_*.pt"))
    if not shard_files:
        print(f"No shards found in {buf}")
        return 0

    rows = []
    for p in shard_files:
        try:
            shard = torch.load(p, map_location="cpu")
            # normalize common schemas
            x = shard.get("x", shard.get("boards"))
            pi = shard.get("pi", shard.get("policies", shard.get("policy", shard.get("probs"))))
            z = shard.get("z", shard.get("values", shard.get("value")))
            if x is None or pi is None or z is None:
                rows.append({"path": str(p), "error": "missing_keys"})
                continue

            stats = analyze_rl_shard({"x": torch.as_tensor(x), "pi": torch.as_tensor(pi), "z": torch.as_tensor(z)})
            stats["path"] = str(p)
            rows.append(stats)
        except Exception as e:
            rows.append({"path": str(p), "error": str(e)})

    bad = []
    if args.min_unique is not None:
        bad = [r for r in rows if isinstance(r.get("unique_ratio"), (int, float)) and r["unique_ratio"] < args.min_unique]

    print(f"Shards: {len(rows)}")
    ok = [r for r in rows if "error" not in r]
    print(f"Ok: {len(ok)} | Errors: {len(rows) - len(ok)}")
    if ok:
        avg_unique = sum(r["unique_ratio"] for r in ok) / len(ok)
        avg_draw = sum(r["z_draw_frac"] for r in ok) / len(ok)
        print(f"Avg unique_ratio: {avg_unique:.3f}")
        print(f"Avg draw_frac:    {avg_draw:.3f}")

    if bad:
        print(f"\nBelow min unique_ratio={args.min_unique}: {len(bad)}")
        for r in bad[:10]:
            print(f"  {r['path']} | n={r.get('n_positions')} unique={r.get('unique_ratio'):.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # collect all keys
        keys = sorted({k for r in rows for k in r.keys()})
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
