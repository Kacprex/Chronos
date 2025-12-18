import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import SHARD_DIR, PHASE1_MODEL_PATH
from src.nn.network import ChessNet


# ==========================
# SETTINGS
# ==========================

BATCH_SIZE = 256
EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = "models/checkpoints/phase1_resume.pt"
os.makedirs("models/checkpoints", exist_ok=True)


# ==========================
# HELPERS
# ==========================

def fmt_time(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def get_shard_paths():
    files = [
        os.path.join(SHARD_DIR, f)
        for f in os.listdir(SHARD_DIR)
        if f.endswith(".pt")
    ]
    files.sort()
    return files


# ==========================
# LOAD CHECKPOINT (if exists)
# ==========================

def load_resume_state(model, optimizer, scaler):
    if not os.path.isfile(CHECKPOINT_PATH):
        log("No checkpoint found, starting fresh.")
        return 0, 0, 0  # epoch, shard, batch

    log(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    return ckpt["epoch"], ckpt["shard"], ckpt["batch"]


# ==========================
# SAVE CHECKPOINT
# ==========================

def save_checkpoint(epoch, shard_idx, batch_idx, model, optimizer, scaler):
    torch.save({
        "epoch": epoch,
        "shard": shard_idx,
        "batch": batch_idx,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, CHECKPOINT_PATH)

    log(f"Checkpoint saved (epoch={epoch}, shard={shard_idx}, batch={batch_idx})")


# ==========================
# TRAINING LOOP (WITH RESUME)
# ==========================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    shards = get_shard_paths()
    total_shards = len(shards)
    log(f"Found {total_shards} shards")

    model = ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


    # Resume state
    resume_epoch, resume_shard, resume_batch = load_resume_state(model, optimizer, scaler)

    loss_policy_fn = torch.nn.CrossEntropyLoss()
    loss_value_fn = torch.nn.MSELoss()

    global_start = time.time()
    total_samples = 0

    for epoch in range(resume_epoch, EPOCHS):
        log(f"=== Epoch {epoch+1}/{EPOCHS} ===")

        for shard_idx in range(resume_shard, total_shards):
            shard_path = shards[shard_idx]

            shard_t0 = time.time()
            shard = torch.load(shard_path, map_location="cpu")

            boards, policies, values = shard["boards"], shard["policies"], shard["values"]
            dataset = TensorDataset(boards, policies, values)

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
            )

            num_batches = len(loader)
            log(f"Loaded shard {shard_idx+1}/{total_shards} ({num_batches} batches).")

            # Determine starting batch (due to resume)
            start_batch = resume_batch if shard_idx == resume_shard else 0

            shard_elapsed = 0
            shard_start = time.time()

            batch_bar = tqdm(
                enumerate(loader),
                total=num_batches,
                desc=f"Shard {shard_idx+1}/{total_shards}",
                dynamic_ncols=True,
            )

            for batch_idx, (b, p, v) in batch_bar:
                if batch_idx < start_batch:
                    continue  # skip already completed batches

                b = b.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)
                v = v.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred_policy, pred_value = model(b)

                    target_idx = p.argmax(1)
                    loss_policy = loss_policy_fn(pred_policy, target_idx)
                    loss_value = loss_value_fn(pred_value, v)
                    loss = loss_policy + loss_value

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_samples += b.size(0)

                # Save checkpoint periodically
                if batch_idx % 200 == 0:
                    save_checkpoint(epoch, shard_idx, batch_idx, model, optimizer, scaler)

                # Live logging
                shard_elapsed = time.time() - shard_start
                progress = (batch_idx + 1) / num_batches
                eta = shard_elapsed / progress - shard_elapsed if progress > 0 else 0

                batch_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "p_loss": f"{loss_policy.item():.4f}",
                    "v_loss": f"{loss_value.item():.4f}",
                    "ETA": fmt_time(eta),
                })

            # End of shard — save checkpoint
            save_checkpoint(epoch, shard_idx + 1, 0, model, optimizer, scaler)

            log(f"Shard {shard_idx+1} finished in {fmt_time(shard_elapsed)}")

        # End of epoch — save final
        torch.save(model.state_dict(), PHASE1_MODEL_PATH)
        log(f"Epoch {epoch+1} complete — model saved.")

        resume_shard = 0  # next epoch starts at shard 0
        resume_batch = 0  

    total_t = time.time() - global_start
    log(f"Training complete. Total samples: {total_samples}. Total time: {fmt_time(total_t)}")


if __name__ == "__main__":
    train()
