# Chronos Project Library

This document is a **living reference** for the Chronos codebase: what each module does, how data flows through the system, and where the important knobs live.

---

## 1) Key ideas (one screen summary)

Chronos follows the usual AlphaZero loop:

1. **Self-play**: play games using **MCTS guided by the neural network**.
2. **Buffer**: store (state, MCTS policy, result) samples to disk as `.pt` shards.
3. **RL training**: train the network on those shards.
4. **Promotion**: pit `latest_model.pth` against `best_model.pth` and promote if the score is high enough.

---

## 2) Repository map

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

## 3) Configuration (`src/config.py`)

The project is intentionally configured from **one file**.

### Paths

- `ENGINE_PATH`: Stockfish executable path (used in Stockfish-vs-AI modes / evaluation features).
- `MODEL_DIR`: directory that stores models.
- `BEST_MODEL_PATH`: the current best model checkpoint.
- `LATEST_MODEL_PATH`: the checkpoint written by training.
- `CHECKPOINT_DIR`: where resuming checkpoints live (e.g. `rl_resume.pt`).

### RL buffer / storage (tuned for ~250 GB spare)

- `RL_BUFFER_DIR`: where self-play writes shards (`.pt` files).
- `RL_BUFFER_MAX_GB`: max size of buffer on disk (default: **250 GB**).
- `RL_BUFFER_PRUNE_KEEP_GB`: after pruning, keep the buffer under this (default: **220 GB**).

**How pruning works:** self-play calls `_prune_rl_buffer_if_needed()` before writing shards. It deletes the oldest shard files until the buffer drops under the keep threshold.

### Discord

- `DISCORD_PROMOTION_WEBHOOK`: webhook URL for promotion notifications.

You can set the webhook without editing code using the environment variable:

```powershell
$env:CHRONOS_PROMOTION_WEBHOOK = "<your webhook url>"
```

If Discord replies with `403 Forbidden`, the webhook token is invalid/revoked or the webhook is no longer allowed to post in that channel.

---

## 4) Data formats

### 4.1 RL shards (`.pt` files)

Self-play saves a shard as a PyTorch dictionary:

- `x`: board tensor, shape `(N, 18, 8, 8)` (float32)
- `pi`: MCTS move distribution, shape `(N, MOVE_SPACE)` (float32)
- `z`: final game result from the current player’s perspective, shape `(N, 1)` (float32)

Where:
- `N` is the number of positions in the shard.
- `MOVE_SPACE` is defined in `src/nn/encoding.py`.

#### RL shard filenames

`RL_Shard_{generation}_{YYYYMMDD}_{HHMMSS}_{loop}_{shard}.pt`

- `generation`: starts at **0**, increments **only on promotion**.
- `loop`: RL loop index from hub option 8 (`1..num_loops`).
- `shard`: monotonically increasing counter (`1..∞`).

This makes it easy to sort shards and identify which generation produced them.

#### Backward compatibility

`train_rl.py` supports both this current format (`x/pi/z`) and an older naming (`boards/policies/values`).

### 4.2 Model checkpoints (`.pth`)

- Stored with `torch.save(model.state_dict(), path)`.
- The architecture must match `src/nn/model.py`.


#### Model history (rollback)

On each successful promotion, Chronos also archives the promoted checkpoint as `models/model_{generation}.pth` and keeps the **last 5** archived generations. This provides a simple rollback mechanism if performance collapses in later generations.

### 4.3 Resume checkpoint (`rl_resume.pt`)

Used to resume RL training and typically includes:
- model weights
- optimizer state
- scaler state (when AMP is enabled)
- training step counters

Exact fields are defined by `src/training/train_rl.py`.

---

## 5) Core modules

### 5.1 Neural network (`src/nn/model.py`)

Defines the policy/value network.

Inputs:
- encoded board tensor from `encoding.encode_board()`

Outputs:
- `policy_logits`: logits over `MOVE_SPACE`
- `value`: scalar in `[-1, 1]` (tanh or similar)

### 5.2 Board & move encoding (`src/nn/encoding.py`)

- `encode_board(board)` builds the **18-plane** representation.
- Move indexing maps chess moves to a fixed action space `MOVE_SPACE`.

**Important limitation (current):** promotion moves are indexed using `from_square` and promotion piece without encoding the `to_square`. That means multiple different promotions can collide into the same index, and `index_to_move()` can become ambiguous for promotions.

En passant is currently not encoded.

### 5.3 MCTS (`src/mcts/mcts.py`, `src/mcts/node.py`)

Runs MCTS with PUCT-style selection:
- Expand: evaluate leaf with the NN → policy prior + value
- Backup: propagate value up the tree
- Final move selection: proportional to visit counts (with optional Dirichlet noise for exploration)

The MCTS class supports **batched inference** for speed.

### 5.4 Self-play (`src/selfplay/self_play_worker.py`)

Responsibilities:
- Spawn multiple worker processes.
- Each worker plays games, collects samples, and periodically writes shards.
- Prune the RL buffer when it exceeds the configured size.

Key tunables (passed from `hub.py`):
- number of workers
- MCTS simulations per move
- inference batch size
- shard size (positions per shard)

### 5.5 RL training (`src/training/train_rl.py`)

Responsibilities:
- Load shards from `RL_BUFFER_DIR`.
- Build a dataset of `(x, pi, z)` samples.
- Train the network for the requested number of epochs/steps.
- Save:
  - `latest_model.pth`
  - resume checkpoint (`rl_resume.pt`)

Losses:
- **Policy loss**: cross-entropy against the target distribution `pi`.
- **Value loss**: MSE against `z`.

AMP (automatic mixed precision) is enabled when CUDA is available.

### 5.6 Promotion (`src/evaluation/promotion.py`)

- Plays `N` games of `latest` vs `best` (switching colors).
- Computes score as: win=1, draw=0.5, loss=0.
- Promotes if `score / N >= threshold`.

If `DISCORD_PROMOTION_WEBHOOK` is configured, it posts the result to Discord.

### 5.7 Discord webhooks (`src/logging/discord_webhooks.py`)

A tiny helper that POSTs JSON to Discord using the standard library (`urllib`).

---

## 6) Typical workflows

### 6.1 First run (minimal)

1. Configure paths in `src/config.py` (Stockfish, models dir, RL buffer dir).
2. Make sure `best_model.pth` exists at `BEST_MODEL_PATH`.
3. Start the hub and run an RL loop:

```powershell
python hub.py
```

Choose option **8** (RL loop) and start with conservative settings.

### 6.2 Storage management

The RL buffer can grow fast. With the defaults:
- Hard cap: **250 GB**
- Prune target: **220 GB**

If you want to free space immediately, you can delete the oldest shard files in `RL_BUFFER_DIR`.


**Note on resuming after manual deletes:** if you manually delete shard files, the RL resume checkpoint (`rl_resume.pt`) can reference shard indices that no longer exist. Chronos will ignore/clear the resume state when it detects missing shards so training can continue without getting stuck.

---

## 7) Troubleshooting cheatsheet

- `KeyError: 'boards'` during RL training → your shards are `x/pi/z` and you need the updated `train_rl.py`.
- `HTTP Error 403: Forbidden` from Discord → webhook invalid/revoked (regenerate).
- Self-play worker crash → most often:
  - missing / wrong `BEST_MODEL_PATH`
  - checkpoint architecture mismatch
  - out-of-memory (reduce workers, batch size, or simulations)

---

## 8) What to improve next

High-impact next steps (not implemented here, but worth tracking):

1. Fix promotion move-index collisions for promotions (include `to_square` in the encoding).
2. Add en passant plane to the board encoding.
3. Make RL training sample a *balanced mix* of old + recent shards (instead of “load everything”).
4. Add an evaluation Elo ladder vs multiple past bests (to reduce regressions).
