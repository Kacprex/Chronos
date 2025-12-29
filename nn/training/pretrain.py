from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn.models.chronos_cnn import ChronosCNN
from nn.data.shards_dataset import ShardsJsonlDataset


def default_root() -> Path:
    return Path(os.environ.get("CHRONOS_DATA_ROOT", r"E:/chronos"))


@dataclass
class TrainConfig:
    shards_jsonl: Path
    out_dir: Path
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def write_event(log_path: Path, obj: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="Path to shards JSONL (usually under E:/chronos/shards/sl/...)")
    ap.add_argument("--run", default="", help="Run id (folder name). Default: timestamp.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = default_root()
    run_id = args.run or f"pretrain_{int(time())}"
    out_dir = root / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = root / "logs" / "events.jsonl"
    metrics_csv = out_dir / "metrics.csv"
    ckpt_path = out_dir / "latest_model.pt"

    cfg = TrainConfig(
        shards_jsonl=Path(args.shards),
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )

    write_event(log_path, {"ts_ms": int(time()*1000), "type": "pretrain_start", "run_id": run_id, "shards": str(cfg.shards_jsonl)})

    ds = ShardsJsonlDataset(cfg.shards_jsonl)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    model = ChronosCNN(in_planes=25).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    mse = torch.nn.MSELoss()

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["epoch", "loss", "loss_value", "loss_pressure", "loss_volatility", "loss_complexity"])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total = 0.0
            lv = lp = lvol = lc = 0.0
            n = 0

            for batch in tqdm(dl, desc=f"epoch {epoch}/{cfg.epochs}"):
                x = batch["x"].to(cfg.device, non_blocking=True)
                y = {k: v.to(cfg.device, non_blocking=True) for k, v in batch["y"].items()}

                out = model(x)

                loss_v = mse(out["value"], y["value"])
                loss_p = mse(out["pressure"], y["pressure"])
                loss_vol = mse(out["volatility"], y["volatility"])
                loss_c = mse(out["complexity"], y["complexity"])

                loss = loss_v + loss_p + 0.5 * loss_vol + loss_c

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss.detach().cpu())
                lv += float(loss_v.detach().cpu())
                lp += float(loss_p.detach().cpu())
                lvol += float(loss_vol.detach().cpu())
                lc += float(loss_c.detach().cpu())
                n += 1

            total /= max(1, n)
            lv /= max(1, n)
            lp /= max(1, n)
            lvol /= max(1, n)
            lc /= max(1, n)

            wcsv.writerow([epoch, total, lv, lp, lvol, lc])
            f.flush()

            write_event(log_path, {
                "ts_ms": int(time()*1000),
                "type": "pretrain_epoch_end",
                "run_id": run_id,
                "epoch": epoch,
                "loss": total,
                "loss_value": lv,
                "loss_pressure": lp,
                "loss_volatility": lvol,
                "loss_complexity": lc,
            })

            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

    write_event(log_path, {"ts_ms": int(time()*1000), "type": "pretrain_done", "run_id": run_id, "ckpt": str(ckpt_path)})

    print(f"Done. Checkpoints: {ckpt_path}")
    print(f"Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
