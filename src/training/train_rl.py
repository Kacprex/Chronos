import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
import time
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.config import RL_BUFFER_DIR, LATEST_MODEL_PATH, BEST_MODEL_PATH
from src.nn.network import ChessNet


def _pick_first(d: dict, keys):
    """Return the first existing value for any of the provided keys."""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(
        f"RL shard missing expected keys. Looked for {keys}. Present keys: {list(d.keys())}"
    )


BATCH_SIZE = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 1           # loop over RL shards
TEMPERATURE_PLIES = 10
INITIAL_TEMPERATURE = 1.25
CHECKPOINT_PATH = "models/checkpoints/rl_resume.pt"

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)


def fmt_time(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def get_rl_shards():
    files = [
        os.path.join(RL_BUFFER_DIR, f)
        for f in os.listdir(RL_BUFFER_DIR)
        if f.startswith("rl_shard_") and f.endswith(".pt")
    ]
    files.sort()
    return files


def load_resume_state(model, optimizer, scaler):
    if not os.path.isfile(CHECKPOINT_PATH):
        log("No RL checkpoint found, starting RL training fresh.")
        return 0, 0, 0  # epoch, shard, batch

    log(f"Loading RL checkpoint from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["shard"], ckpt["batch"]


def save_checkpoint(epoch, shard_idx, batch_idx, model, optimizer, scaler):
    torch.save({
        "epoch": epoch,
        "shard": shard_idx,
        "batch": batch_idx,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, CHECKPOINT_PATH)

    log(f"RL checkpoint saved at epoch={epoch}, shard={shard_idx}, batch={batch_idx}")


def train_rl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"RL training using device: {device}")

    shards = get_rl_shards()
    if not shards:
        log("No RL shards found. Run self-play first.")
        return

    total_shards = len(shards)
    log(f"Found {total_shards} RL shard(s) in {RL_BUFFER_DIR}")

    model = ChessNet().to(device)
    # Start from latest model or best model
    if os.path.isfile(LATEST_MODEL_PATH):
        log(f"Loading latest model from {LATEST_MODEL_PATH}")
        model.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))
    elif os.path.isfile(BEST_MODEL_PATH):
        log(f"No latest model, using best model from {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    loss_policy_fn = torch.nn.KLDivLoss(reduction="batchmean")
    loss_value_fn = torch.nn.MSELoss()

    resume_epoch, resume_shard, resume_batch = load_resume_state(model, optimizer, scaler)

    global_start = time.time()
    total_samples = 0

    for epoch in range(resume_epoch, EPOCHS):
        log(f"=== RL Epoch {epoch+1}/{EPOCHS} ===")

        for shard_idx in range(resume_shard, total_shards):
            shard_path = shards[shard_idx]
            shard_t0 = time.time()

            shard = torch.load(shard_path, map_location="cpu")

            # Support both shard schemas:
            # - old:  {"boards", "policies", "values"}
            # - new:  {"x", "pi", "z"} (self-play buffer)
            boards = _pick_first(shard, ["boards", "x"])
            policies = _pick_first(shard, ["policies", "pi", "probs"])
            values = _pick_first(shard, ["values", "z", "value"])

            boards = torch.as_tensor(boards)
            policies = torch.as_tensor(policies)
            values = torch.as_tensor(values, dtype=torch.float32)
            if values.ndim == 1:
                values = values.view(-1, 1)

            dataset = TensorDataset(boards, policies, values)
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=True,
                num_workers=6,          # Ryzen sweet spot
                persistent_workers=True,
                prefetch_factor=2,
            )
            

            num_batches = len(loader)
            log(f"Loaded RL shard {shard_idx+1}/{total_shards} ({len(dataset)} samples, {num_batches} batches).")

            start_batch = resume_batch if shard_idx == resume_shard else 0
            shard_start = time.time()

            batch_bar = tqdm(
                enumerate(loader),
                total=num_batches,
                desc=f"RL Shard {shard_idx+1}/{total_shards}",
                dynamic_ncols=True,
            )

            for batch_idx, (b, pi_target, v_target) in batch_bar:
                if batch_idx < start_batch:
                    continue

                b = b.to(device, non_blocking=True)
                pi_target = pi_target.to(device, non_blocking=True)
                v_target = v_target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, v_pred = model(b)
                    log_probs = torch.log_softmax(logits, dim=1)
                    pi_target_norm = pi_target / (pi_target.sum(dim=1, keepdim=True) + 1e-8)

                    policy_loss = loss_policy_fn(log_probs, pi_target_norm)
                    value_loss = loss_value_fn(v_pred, v_target)
                    loss = policy_loss + value_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_samples += b.size(0)

                if batch_idx % 200 == 0:
                    save_checkpoint(epoch, shard_idx, batch_idx, model, optimizer, scaler)

                shard_elapsed = time.time() - shard_start
                progress = (batch_idx + 1) / num_batches
                shard_eta = shard_elapsed / progress - shard_elapsed if progress > 0 else 0

                batch_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "p_loss": f"{policy_loss.item():.4f}",
                    "v_loss": f"{value_loss.item():.4f}",
                    "ETA": fmt_time(shard_eta),
                })

            save_checkpoint(epoch, shard_idx + 1, 0, model, optimizer, scaler)
            log(f"RL shard {shard_idx+1} finished in {fmt_time(time.time() - shard_start)}")

        # End epoch → save latest model
        os.makedirs(os.path.dirname(LATEST_MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), LATEST_MODEL_PATH)
        log(f"Saved latest RL model to {LATEST_MODEL_PATH}")

        resume_shard = 0
        resume_batch = 0

    total_time = time.time() - global_start
    log(f"✔ RL training complete. samples={total_samples}, time={fmt_time(total_time)}")


if __name__ == "__main__":
    train_rl()
