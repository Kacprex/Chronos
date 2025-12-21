# Chronos

Chronos is an AlphaZero-style chess project:

- **Self-play** generates training data using **MCTS + a neural network**.
- **RL training** learns from those self-play positions (**policy + value heads**).
- **Promotion** evaluates `latest_model.pth` vs `best_model.pth` and promotes if the winrate clears a threshold.

The main entrypoint is **`hub.py`** (interactive menu).

---

## Project layout

```
chronos/
  hub.py
  README.md
  docs/
    PROJECT_LIBRARY.md
  src/
    config.py
    evaluation/
      promotion.py
    logging/
      discord_webhooks.py
    mcts/
      mcts.py
      node.py
    nn/
      encoding.py
      model.py
    selfplay/
      self_play_worker.py
    training/
      train_rl.py
```

---

## Installation (Windows + PowerShell)

Create a virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

1) **Install PyTorch** (CPU or CUDA) using the command generated on the PyTorch website.

2) Install the remaining packages:

```powershell
pip install -r requirements.txt
```

---

## Configuration

All default settings live in `src/config.py`.

### Required paths

- `ENGINE_PATH` – path to your Stockfish binary.
- `MODELS_DIR` – directory containing `best_model.pth` and `latest_model.pth`.

### RL buffer & disk usage

Self-play writes shards to `RL_BUFFER_DIR` (default: `E:/chronos/chronos_rl_buffer`).

The buffer is auto-pruned when it gets too large:

- `RL_BUFFER_MAX_GB` (default **250**) – hard cap.
- `RL_BUFFER_PRUNE_KEEP_GB` (default **220**) – after pruning, keep roughly this much.

This matches the assumption that you have **~250 GB** of storage to spare for RL data.

### Discord notifications (promotion)

Promotion results can be posted to Discord.

Set the webhook either by editing `DISCORD_PROMOTION_WEBHOOK` in `src/config.py` **or** by setting an environment variable:

```powershell
$env:CHRONOS_PROMOTION_WEBHOOK = "<your webhook url>"
```

If you get `HTTP Error 403: Forbidden`, the webhook URL/token is invalid or has been revoked—create a new webhook and update it.

---

## Running

```powershell
python hub.py
```

Recommended flow:

- **Option 8: Run RL loop** (self-play → train_rl → promote)

The RL loop asks for:
- self-play games per iteration
- number of self-play worker processes
- MCTS simulations per move
- RL shard size (positions)
- promotion match games + threshold

---

## RL shard format (important)

Self-play saves each shard as a PyTorch `.pt` dict with:

- `x`: board tensor, shape `(N, 18, 8, 8)`
- `pi`: MCTS move distribution per position, shape `(N, MOVE_SPACE)`
- `z`: game result from the current player’s perspective, shape `(N, 1)`

`train_rl.py` supports both the current keys (`x/pi/z`) and an older naming (`boards/policies/values`) if you have older shards.

---

## Known limitations

- **Promotion move indexing collision:** promotions are currently indexed without encoding the `to_square`, so different promotion moves from the same `from_square` can collide. This can make `index_to_move` ambiguous for promotions.
- **En passant** is not encoded.

(These are documented in more detail in `docs/PROJECT_LIBRARY.md`.)

---

## Troubleshooting

### `KeyError: 'boards'` during RL training

Your RL shards were written with the new keys (`x/pi/z`) but your `train_rl.py` expected `boards/policies/values`.
Update `src/training/train_rl.py` (this repo version already handles both formats).

### Self-play workers crash

- Make sure `best_model.pth` exists where `BEST_MODEL_PATH` points.
- If you changed the model architecture, ensure the checkpoint matches the code.

### Discord webhook errors

- `403 Forbidden`: invalid / revoked webhook token, regenerate.
- `404 Not Found`: wrong webhook URL.
- Corporate proxies/firewalls can also block webhook posts.

