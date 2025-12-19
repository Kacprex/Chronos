# ♟️ Chronos

Chronos is an **AlphaZero-style chess AI** implemented in Python:
- a **policy+value neural network** (`ChessNet`)
- **MCTS** guided by the network
- **Supervised Learning (SL)** from strong human games (sharded dataset)
- **Reinforcement Learning (RL)** from self-play (MCTS targets)
- evaluation utilities (diversity test, promotion matches)

> The project is built around **shards** (`.pt` files) so training can run on consumer RAM and resume safely.

---

## Requirements

- Python 3.10+ (tested with 3.11)
- Optional: NVIDIA GPU + CUDA (PyTorch)

Python deps:
```bash
pip install torch numpy pandas tqdm python-chess
```

---

## Configuration

Edit **`src/config.py`** to match your machine:

- `GM_GAMES_PATH` — CSV with `pgn` + `Result`
- `SHARD_DIR` — output dir for SL shards (`shard_*.pt`)
- `RL_BUFFER_DIR` — output dir for RL shards (`rl_shard_*.pt`)
- `ENGINE_PATH` — Stockfish binary (optional)
- `BEST_MODEL_PATH`, `LATEST_MODEL_PATH` — promotion workflow paths

---

## Data format (what training expects)

### SL shard (`preprocess_rust.py`)
Saved as:
- `boards`: `(N, 18, 8, 8)` float32
- `policies`: `(N, 4672)` float32 **one-hot** (the next move from the PGN)
- `values`: `(N, 1)` float32 in `[-1, 1]` (game outcome from White POV)

### RL shard (`src/selfplay/self_play_worker.py`)
Saved as:
- `boards`: `(N, 18, 8, 8)` float32
- `policies`: `(N, 4672)` float32 **probabilities** from MCTS
- `values`: `(N, 1)` float32 in `[-1, 1]` (terminal outcome, perspective-correct per position)

---

## Phase 1: Preprocess GM games into SL shards

```bash
python preprocess_rust.py
```

Your dataset must include:
- `pgn` (full PGN text)
- `Result` (`1-0`, `0-1`, `1/2-1/2`)

---

## Phase 2: Supervised training

```bash
python -m src.training.train_supervised
```

- resume checkpoint: `models/checkpoints/phase1_resume.pt`
- output model: `PHASE1_MODEL_PATH` (from `src/config.py`)

---

## Phase 3+: RL loop (self-play → train_rl → promote)

### 1) Seed best/latest from SL
After SL, copy the SL model to both “best” and “latest”:

```powershell
# Windows PowerShell (adjust paths)
copy models\checkpoints\sl_final.pth models\best_model.pth
copy models\checkpoints\sl_final.pth models\latest_model.pth
```

### 2) Generate self-play RL shards
```bash
python -m src.selfplay.self_play_worker
```

### 3) Train RL on RL buffer
```bash
python -m src.training.train_rl
```

### 4) Evaluate and promote
```bash
python -m src.evaluation.promotion
```

---

## hub.py (CLI)

```bash
python hub.py
```

The hub provides a menu to run:
- self-play generation
- RL training
- AI-vs-AI / Stockfish-vs-AI games (PGNs)
- diversity test
- promotion evaluation

---

## Known issues / TODOs

1. `src/selfplay/self_play_worker.py` currently contains a **badly-indented import line** (merge artifact) and may raise `IndentationError` until fixed.
2. `src/evaluation/stockfish_eval.py` is **experimental / unfinished** (path/import inconsistencies).
3. `src/mcts/node.py` and `src/mcts/puct.py` are placeholders (unused by the active MCTS).

If you want, I can generate another “fix zip” that cleans these up.

---

## Repo hygiene (don’t push huge datasets)

If you keep a large dataset folder in repo root (e.g. `./data/`) **do not commit it**.

Suggested `.gitignore` pattern:
```gitignore
/data/
!/src/data/
```
