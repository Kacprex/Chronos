from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from common.paths import ensure_layout
from common.encoding import encode_fen_18x8x8, INPUT_DIM
from common.dataset_io import write_dataset_bin
from common.logging_utils import log_line
from common.public_export import export_public_status

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_labeled_jsonl_gz", type=str, required=True)
    ap.add_argument("--out_dataset_bin", type=str, required=True)
    ap.add_argument("--label_scale_cp", type=float, default=600.0, help="cp/scale -> [-1,1]")
    ap.add_argument("--max_records", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    p = ensure_layout()
    log_path = p["logs"] / "build_dataset.log"
    log_line(log_path, f"START build_dataset scale={args.label_scale_cp}")

    in_path = Path(args.in_labeled_jsonl_gz)
    out_path = Path(args.out_dataset_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: count
    n = 0
    with gzip.open(in_path, "rt", encoding="utf-8") as gz:
        for _ in gz:
            n += 1
            if args.max_records and n >= args.max_records:
                break

    X = np.zeros((n, INPUT_DIM), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)

    export_public_status({"phase": "SL_BUILD_DATASET", "total": n, "done": 0})

    i = 0
    with gzip.open(in_path, "rt", encoding="utf-8") as gz:
        for line in tqdm(gz, total=n, desc="Building dataset"):
            if args.max_records and i >= args.max_records:
                break
            rec = json.loads(line)
            fen = rec["fen"]
            cp = rec.get("cp", None)
            if cp is None:
                continue
            X[i, :] = encode_fen_18x8x8(fen)
            yv = clamp(float(cp) / float(args.label_scale_cp), -1.0, 1.0)
            y[i] = yv
            i += 1
            if i % 2000 == 0:
                export_public_status({"phase": "SL_BUILD_DATASET", "total": n, "done": i})

    # If some were skipped
    if i != n:
        X = X[:i]
        y = y[:i]
        n = i

    write_dataset_bin(out_path, X, y)
    log_line(log_path, f"DONE build_dataset out={out_path} n={n}")
    export_public_status({"phase": "IDLE", "last": "build_dataset", "out": str(out_path), "n": n})

if __name__ == "__main__":
    main()
