from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nn.models.chronos_hybrid import ChronosHybridNet
from nn.move_index import MOVE_SPACE
from rl.util import default_root, append_event, now_ms


class PolicyJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offsets: list[int] = []
        self._build()

    def _build(self) -> None:
        self._offsets.clear()
        with self.path.open("rb") as f:
            off = 0
            for line in f:
                self._offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = self._offsets[idx]
        with self.path.open("rb") as f:
            f.seek(off)
            line = f.readline()
        obj = json.loads(line.decode("utf-8"))

        planes = np.asarray(obj["planes"], dtype=np.float32).reshape(25, 8, 8)

        t = obj.get("targets") or {}
        y_val = float(obj.get("value_target", t.get("value", 0.0)))
        y_p = float(t.get("pressure", 0.0))
        y_vol = float(t.get("volatility", 0.0))
        y_c = float(t.get("complexity", 0.0))

        pol = obj.get("policy") or {"idx": [], "p": []}
        idxs = pol.get("idx") or []
        ps = pol.get("p") or []
        if len(idxs) != len(ps):
            idxs, ps = [], []

        wgt = float(obj.get("weight", 1.0))
        return {
            "x": torch.from_numpy(planes),
            "y_value": torch.tensor([y_val], dtype=torch.float32),
            "y_pressure": torch.tensor([y_p], dtype=torch.float32),
            "y_volatility": torch.tensor([y_vol], dtype=torch.float32),
            "y_complexity": torch.tensor([y_c], dtype=torch.float32),
            "policy_idx": torch.tensor(idxs, dtype=torch.int64),
            "policy_p": torch.tensor(ps, dtype=torch.float32),
            "w": torch.tensor([wgt], dtype=torch.float32),
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)

    y_value = torch.cat([b["y_value"] for b in batch], dim=0).view(-1, 1)
    y_pressure = torch.cat([b["y_pressure"] for b in batch], dim=0).view(-1, 1)
    y_vol = torch.cat([b["y_volatility"] for b in batch], dim=0).view(-1, 1)
    y_comp = torch.cat([b["y_complexity"] for b in batch], dim=0).view(-1, 1)

    maxk = max((b["policy_idx"].numel() for b in batch), default=0)
    idx_pad = torch.full((len(batch), maxk), -1, dtype=torch.int64)
    p_pad = torch.zeros((len(batch), maxk), dtype=torch.float32)
    for i, b in enumerate(batch):
        k = b["policy_idx"].numel()
        if k > 0:
            idx_pad[i, :k] = b["policy_idx"]
            p_pad[i, :k] = b["policy_p"]

    wgt = torch.cat([b["w"] for b in batch], dim=0).view(-1, 1)
    return {"x": x, "y_value": y_value, "y_pressure": y_pressure, "y_vol": y_vol, "y_comp": y_comp, "idx": idx_pad, "p": p_pad, "w": wgt}


def policy_loss(logits: torch.Tensor, idx: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=1)
    mask = (idx >= 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    idx2 = idx.clamp(min=0)
    gathered = torch.gather(logp, 1, idx2)  # [B,K]
    loss = -(p * gathered) * mask.float()
    denom = (p * mask.float()).sum(dim=1).clamp(min=1e-6)
    per = loss.sum(dim=1) / denom
    return per.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with planes + policy (either labeled.jsonl with --multipv, or az_selfplay.jsonl)")
    ap.add_argument("--run", default="", help="Run id. Default: hybrid_<timestamp>")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = default_root()
    run_id = args.run or f"hybrid_{int(time())}"
    out_dir = root / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = out_dir / "hybrid_metrics.csv"
    ckpt_path = out_dir / "hybrid_latest_model.pt"

    ds = PolicyJsonlDataset(args.data)
    dl = DataLoader(ds, batch_size=int(args.batch), shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChronosHybridNet(in_planes=25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    mse = torch.nn.MSELoss(reduction="none")

    append_event(root, {"ts_ms": now_ms(), "type": "hybrid_train_start", "run_id": run_id, "data": str(args.data)})

    with metrics.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["epoch", "loss", "loss_policy", "loss_value", "loss_pressure", "loss_volatility", "loss_complexity"])

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            tot = lp = lv = lpr = lvol = lc = 0.0
            n = 0

            for batch in tqdm(dl, desc=f"hybrid epoch {epoch}/{args.epochs}"):
                x = batch["x"].to(device, non_blocking=True)
                idx = batch["idx"].to(device, non_blocking=True)
                pp = batch["p"].to(device, non_blocking=True)
                wgt = batch["w"].to(device, non_blocking=True)

                yv = batch["y_value"].to(device, non_blocking=True)
                yp = batch["y_pressure"].to(device, non_blocking=True)
                yvol = batch["y_vol"].to(device, non_blocking=True)
                yc = batch["y_comp"].to(device, non_blocking=True)

                out = model(x)

                l_pol = policy_loss(out["policy"], idx, pp)
                l_v = (mse(out["value"], yv) * wgt).mean()
                l_p = (mse(out["pressure"], yp) * wgt).mean()
                l_vo = (mse(out["volatility"], yvol) * wgt).mean()
                l_c = (mse(out["complexity"], yc) * wgt).mean()

                loss = l_pol + l_v + l_p + 0.5 * l_vo + l_c

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tot += float(loss.detach().cpu())
                lp += float(l_pol.detach().cpu())
                lv += float(l_v.detach().cpu())
                lpr += float(l_p.detach().cpu())
                lvol += float(l_vo.detach().cpu())
                lc += float(l_c.detach().cpu())
                n += 1

            tot /= max(1, n)
            lp /= max(1, n)
            lv /= max(1, n)
            lpr /= max(1, n)
            lvol /= max(1, n)
            lc /= max(1, n)

            wcsv.writerow([epoch, tot, lp, lv, lpr, lvol, lc])
            f.flush()

            torch.save({"kind": "hybrid_policy", "model": model.state_dict(), "epoch": epoch}, ckpt_path)
            append_event(root, {"ts_ms": now_ms(), "type": "hybrid_train_epoch_end", "run_id": run_id, "epoch": epoch, "loss": tot, "loss_policy": lp})

    append_event(root, {"ts_ms": now_ms(), "type": "hybrid_train_done", "run_id": run_id, "ckpt": str(ckpt_path), "metrics": str(metrics)})
    print(f"Done. CKPT: {ckpt_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
