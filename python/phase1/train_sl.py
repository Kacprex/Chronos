"""Supervised learning trainer (Phase 1) for Chronos.

This version is tuned for:
- **low RAM usage** on Windows (no fancy-indexing copies from memmap)
- **live metrics** compatible with Streamlit dashboard
- fast, append-only log files

Key changes vs older versions:
- Training scans the memmap in contiguous slices (batch = slice), avoiding huge temporary NumPy arrays.
- Optional per-epoch chunk shuffling (still uses contiguous slices) to introduce randomness without killing RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Helpers
# -----------------------------

def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_log(log_path: Path, msg: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(msg)
        if not msg.endswith("\n"):
            f.write("\n")


@dataclass
class DatasetMemmap:
    X: np.memmap
    y: np.memmap
    n: int
    input_dim: int


def load_dataset_memmap(dataset_dir: Path, *, log_path: Path) -> DatasetMemmap:
    """Load dataset from data/dataset/{X.bin,y.bin,meta.json} as memmaps."""
    meta_path = dataset_dir / "meta.json"
    x_path = dataset_dir / "X.bin"
    y_path = dataset_dir / "y.bin"

    if not meta_path.exists() or not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing dataset files. Expected: {meta_path}, {x_path}, {y_path}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n = int(meta["n"]) 
    input_dim = int(meta["input_dim"]) 
    x_dtype = np.dtype(meta.get("X_dtype", "uint8"))
    y_dtype = np.dtype(meta.get("y_dtype", "float32"))

    append_log(log_path, f"[{ts()}] Loading dataset memmaps")
    append_log(log_path, f"[{ts()}]  n={n:,} input_dim={input_dim} X_dtype={x_dtype} y_dtype={y_dtype}")

    X = np.memmap(x_path, mode="r", dtype=x_dtype, shape=(n, input_dim))
    y = np.memmap(y_path, mode="r", dtype=y_dtype, shape=(n,))

    return DatasetMemmap(X=X, y=y, n=n, input_dim=input_dim)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def write_metrics_row(metrics_csv: Path, row: dict) -> None:
    ensure_dir(metrics_csv.parent)
    file_exists = metrics_csv.exists()
    with metrics_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=str, default=str(Path("data") / "dataset"))
    p.add_argument("--out_model", type=str, default=str(Path("data") / "models" / "sl_value.pth"))

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--val_split", type=float, default=0.01)
    p.add_argument(
        "--shuffle_chunk_size",
        type=int,
        default=262144,
        help="If >0, shuffle chunk order each epoch (keeps contiguous slices). 0 = fully sequential.",
    )
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--log_name", type=str, default="train_sl.log")
    p.add_argument("--metrics_name", type=str, default="train_sl_metrics.csv")
    p.add_argument("--status_name", type=str, default="train_sl_status.json")

    p.add_argument(
        "--log_every",
        type=int,
        default=25,
        help="Write a metrics row every N training steps.",
    )

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    return p.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    logs_dir = repo_root / "data" / "logs"
    status_dir = repo_root / "data" / "status"
    metrics_dir = repo_root / "data" / "metrics"

    log_path = logs_dir / args.log_name
    status_path = status_dir / args.status_name
    metrics_csv = metrics_dir / args.metrics_name

    ensure_dir(log_path.parent)
    ensure_dir(status_path.parent)
    ensure_dir(metrics_csv.parent)

    append_log(log_path, f"[{ts()}] === train_sl start ===")
    append_log(log_path, f"[{ts()}] Args: {vars(args)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    append_log(log_path, f"[{ts()}] Device: {device}")

    dataset_dir = Path(args.dataset_dir)
    ds = load_dataset_memmap(dataset_dir, log_path=log_path)

    n = ds.n
    val_n = max(1, int(n * float(args.val_split)))
    train_n = n - val_n

    # contiguous split: [0..train_n) train, [train_n..n) val
    append_log(log_path, f"[{ts()}] Split: train={train_n:,} val={val_n:,} (contiguous)")

    model = ValueNet(ds.input_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # status init
    status = {
        "phase": "sl",
        "running": True,
        "epoch": 0,
        "epochs": int(args.epochs),
        "step": 0,
        "steps_per_epoch": int(math.ceil(train_n / args.batch_size)),
        "train_samples": int(train_n),
        "val_samples": int(val_n),
        "train_loss": None,
        "val_loss": None,
        "updated_at": ts(),
    }
    atomic_write_json(status_path, status)

    rng = np.random.default_rng(args.seed)

    global_step = 0
    out_model = Path(args.out_model)
    ensure_dir(out_model.parent)

    chunk_size = int(args.shuffle_chunk_size)
    if chunk_size < 0:
        chunk_size = 0

    # precompute chunk starts for training range
    if chunk_size == 0:
        chunk_starts = [0]
        chunk_size = train_n
    else:
        chunk_starts = list(range(0, train_n, chunk_size))

    append_log(log_path, f"[{ts()}] Train: batch_size={args.batch_size} chunk_size={chunk_size} chunks={len(chunk_starts)}")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()

        # shuffle chunk order each epoch (keeps contiguous slices => low RAM)
        if len(chunk_starts) > 1 and int(args.shuffle_chunk_size) > 0:
            rng.shuffle(chunk_starts)

        epoch_loss_sum = 0.0
        epoch_count = 0
        t0 = time.time()

        for cs in chunk_starts:
            ce = min(cs + chunk_size, train_n)
            # iterate contiguous batches within the chunk
            for i in range(cs, ce, int(args.batch_size)):
                j = min(i + int(args.batch_size), ce)

                xb_np = ds.X[i:j]  # memmap slice (no big copy)
                yb_np = ds.y[i:j]

                # move to torch
                xb = torch.from_numpy(xb_np).to(device, non_blocking=True).float()
                yb = torch.from_numpy(yb_np).to(device, non_blocking=True).float().view(-1, 1)

                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

                bs = (j - i)
                epoch_loss_sum += float(loss.item()) * bs
                epoch_count += bs

                global_step += 1

                if args.log_every > 0 and (global_step % int(args.log_every) == 0):
                    elapsed = max(1e-9, time.time() - t0)
                    samples_per_s = epoch_count / elapsed
                    train_loss = epoch_loss_sum / max(1, epoch_count)

                    status.update(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "train_loss": train_loss,
                            "updated_at": ts(),
                        }
                    )
                    atomic_write_json(status_path, status)

                    # metrics row
                    write_metrics_row(
                        metrics_csv,
                        {
                            "ts": ts(),
                            "phase": "sl",
                            "epoch": epoch,
                            "step": global_step,
                            "train_loss": train_loss,
                            "val_loss": "" if status.get("val_loss") is None else status["val_loss"],
                            "samples_per_s": f"{samples_per_s:.1f}",
                        },
                    )

                    append_log(
                        log_path,
                        f"[{ts()}] epoch {epoch}/{args.epochs} step {global_step} train_loss={train_loss:.6f} samples/s={samples_per_s:.1f}",
                    )

        # ---- epoch end: validation ----
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_count = 0
            for i in range(train_n, n, int(args.batch_size)):
                j = min(i + int(args.batch_size), n)
                xb_np = ds.X[i:j]
                yb_np = ds.y[i:j]

                xb = torch.from_numpy(xb_np).to(device, non_blocking=True).float()
                yb = torch.from_numpy(yb_np).to(device, non_blocking=True).float().view(-1, 1)

                pred = model(xb)
                loss = loss_fn(pred, yb)
                bs = (j - i)
                val_loss_sum += float(loss.item()) * bs
                val_count += bs

            val_loss = val_loss_sum / max(1, val_count)

        train_loss = epoch_loss_sum / max(1, epoch_count)
        append_log(log_path, f"[{ts()}] epoch {epoch} done: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # status + metrics
        status.update(
            {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "updated_at": ts(),
            }
        )
        atomic_write_json(status_path, status)

        write_metrics_row(
            metrics_csv,
            {
                "ts": ts(),
                "phase": "sl",
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "samples_per_s": "",  # epoch summary row
            },
        )

        # save model
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": ds.input_dim,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "epoch": epoch,
                "global_step": global_step,
            },
            out_model,
        )
        append_log(log_path, f"[{ts()}] Saved model -> {out_model}")

    status["running"] = False
    status["updated_at"] = ts()
    atomic_write_json(status_path, status)
    append_log(log_path, f"[{ts()}] === train_sl end ===")


if __name__ == "__main__":
    main()
