from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn.models.chronos_cnn import ChronosCNN
from rl.data.rl_dataset import RLLabeledJsonlDataset
from rl.util import default_root, append_event, now_ms


@dataclass
class TrainCfg:
    labeled_jsonl: Path
    out_dir: Path
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, help="labeled.jsonl from rl/analyze/label_stockfish.py")
    ap.add_argument("--run", default="", help="Run id. Default: rltrain_<timestamp>")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = default_root()
    run_id = args.run or f"rltrain_{int(time())}"
    out_dir = root / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "rl_metrics.csv"
    ckpt_path = out_dir / "rl_latest_model.pt"

    cfg = TrainCfg(
        labeled_jsonl=Path(args.labeled),
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )

    append_event(root, {"ts_ms": now_ms(), "type": "rl_train_start", "run_id": run_id, "labeled": str(cfg.labeled_jsonl)})

    ds = RLLabeledJsonlDataset(cfg.labeled_jsonl)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    model = ChronosCNN(in_planes=25).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    mse = torch.nn.MSELoss(reduction="none")

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["epoch", "loss", "loss_value", "loss_pressure", "loss_volatility", "loss_complexity"])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total = lv = lp = lvol = lc = 0.0
            n = 0

            for batch in tqdm(dl, desc=f"rl epoch {epoch}/{cfg.epochs}"):
                x = batch["x"].to(cfg.device, non_blocking=True)
                y = {k: v.to(cfg.device, non_blocking=True) for k, v in batch["y"].items()}
                wgt = batch["w"].to(cfg.device, non_blocking=True)  # [B,1]

                out = model(x)

                loss_v = mse(out["value"], y["value"]) * wgt
                loss_p = mse(out["pressure"], y["pressure"]) * wgt
                loss_vol = mse(out["volatility"], y["volatility"]) * wgt
                loss_c = mse(out["complexity"], y["complexity"]) * wgt

                loss = loss_v.mean() + loss_p.mean() + 0.5 * loss_vol.mean() + loss_c.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss.detach().cpu())
                lv += float(loss_v.mean().detach().cpu())
                lp += float(loss_p.mean().detach().cpu())
                lvol += float(loss_vol.mean().detach().cpu())
                lc += float(loss_c.mean().detach().cpu())
                n += 1

            total /= max(1, n)
            lv /= max(1, n)
            lp /= max(1, n)
            lvol /= max(1, n)
            lc /= max(1, n)

            wcsv.writerow([epoch, total, lv, lp, lvol, lc])
            f.flush()

            append_event(root, {
                "ts_ms": now_ms(),
                "type": "rl_train_epoch_end",
                "run_id": run_id,
                "epoch": epoch,
                "loss": total,
                "loss_value": lv,
                "loss_pressure": lp,
                "loss_volatility": lvol,
                "loss_complexity": lc,
            })

            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

    append_event(root, {"ts_ms": now_ms(), "type": "rl_train_done", "run_id": run_id, "ckpt": str(ckpt_path), "metrics": str(metrics_csv)})
    print(f"Done. Checkpoint: {ckpt_path}")
    print(f"Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
