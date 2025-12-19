# Chronos project library

A “library index” of the codebase: **what each file is for, what it exports, and how it’s used**.

This version includes the recent additions:
- **Discord webhook logging** for promotion runs
- **Chronos rating** (Stockfish-based) that runs only when a model is promoted
- RL buffer retention cap (keep last N RL shards)

---

## High-level architecture

1. **Encode** a position → 18×8×8 planes (`src/nn/encoding.py`)
2. **Neural net** → policy logits (4672) + value (-1..1) (`src/nn/network.py`)
3. **MCTS** refines policy (`src/mcts/mcts.py`)
4. **Training**
   - SL: imitate human next-move + game outcome
   - RL: learn from MCTS policy targets + self-play outcome
5. **Promotion**: latest vs best (`src/evaluation/promotion.py`)
6. **Rating** (only after promotion): best vs Stockfish (`src/evaluation/chronos_rating.py`)
7. **Logging**: Discord webhooks (`src/logging/discord_webhooks.py`)

---

## Configuration & environment

### `src/config.py`
Central configuration: paths and output dirs.
- `GM_GAMES_PATH`, `SHARD_DIR`
- `RL_BUFFER_DIR`
- `ENGINE_PATH` (Stockfish)
- `BEST_MODEL_PATH`, `LATEST_MODEL_PATH`
- PGN output paths (`AIVSAI`, `SFVSAI`)

**Discord webhook config (recommended via env vars, not committed):**
- `CHRONOS_PROMOTION_WEBHOOK` → promotion summary embeds
- `CHRONOS_RATING_WEBHOOK` → rating embeds (only when promoted)

**RL buffer retention:**
- `CHRONOS_MAX_RL_SHARDS` (env var) controls how many `rl_shard_*.pt` files are kept.

---

## Neural network & encoding

### `src/nn/encoding.py`
Board encoding + AlphaZero-style move indexing.
- `MOVE_SPACE = 4672`
- `encode_board(board) -> (18,8,8)` float32 planes
- `move_to_index(move) -> int|None` *(or `move_to_index(move, board)` depending on version)*
- `index_to_move(index, board) -> chess.Move` (scans legal moves)

### `src/nn/network.py`
Policy+value CNN.
- `ResidualBlock`
- `ChessNet(channels=128, num_res_blocks=5)`
  - output policy logits: `(B,4672)`
  - output value: `(B,1)` in `[-1,1]`

---

## MCTS

### `src/mcts/mcts.py`
The active MCTS implementation.
- `Node` (internal; stores board, prior, visits, value_sum, children)
- `MCTS.run(board, move_number, add_noise=True) -> (moves, probs)`
  - Uses internal temperature scheduling (explore early, deterministic later)
  - Returns probability distribution derived from child visits

> Note: Old placeholder files like `src/mcts/node.py` and `src/mcts/puct.py` are not used by the active MCTS (and can stay removed).

---

## Self-play & RL data generation

### `src/selfplay/opening_book.py`
Small curated opening set.
- `OPENINGS`
- `play_random_opening(board, max_plies=6) -> int`

### `src/selfplay/encode_game.py`
PGN writing utilities.
- `GameRecord` (moves_uci, result, headers; can append to file safely)
- `build_game_record_from_moves(moves, result, ...) -> GameRecord`

### `src/selfplay/self_play_worker.py`
Self-play + RL shard writer.
- `play_single_game(...) -> (boards, policies, z_white)`
- `self_play(num_games, simulations, shard_size)`
  - Loads `BEST_MODEL_PATH`
  - Writes `rl_shard_*.pt` to `RL_BUFFER_DIR`

**RL shard format** (`rl_shard_*.pt`):
- `boards`: `(N, 18, 8, 8)` float32
- `policies`: `(N, 4672)` float32 probabilities from MCTS
- `values`: `(N, 1)` float32 in `[-1, 1]` from **side-to-move perspective**

**Disk safety**:
- Keeps only newest `CHRONOS_MAX_RL_SHARDS` RL shards (default 300).

---

## Training

### `src/training/train_supervised.py`
Supervised training from SL shards.
- resume checkpoint: `models/checkpoints/phase1_resume.pt`
- key functions: `get_shard_paths`, `load_resume_state`, `save_checkpoint`, `train`

Loss:
- policy: `CrossEntropyLoss(policy_logits, argmax(onehot_policy))`
- value: `MSELoss(value_pred, value_target)`

### `src/training/train_rl.py`
RL training from RL shards.
- resume checkpoint: `models/checkpoints/rl_resume.pt`
- key functions: `get_rl_shards`, `load_resume_state`, `save_checkpoint`, `train_rl`

Loss:
- policy: `KLDivLoss(log_softmax(logits), target_pi)`
- value: `MSELoss(value_pred, value_target)`

Output:
- updates `LATEST_MODEL_PATH`

---

## Evaluation, promotion, and rating

### `src/evaluation/promotion.py`
Promotion check: latest vs best.
- `evaluate_and_promote(num_games=50, threshold=0.55, ...)`
  - alternates colors
  - promotes if average score ≥ threshold
  - emits **Discord webhook embed** summarizing winrate & outcome

### `src/evaluation/chronos_rating.py`
Runs only when a promotion happens.
- Plays a short match: Chronos (best) vs Stockfish (depth configured)
- Computes:
  - match score
  - approximate Elo difference vs that Stockfish config
  - “Chronos rating index” = 1500 + EloDiff (internal metric)
- Sends embed to rating webhook.

### `src/evaluation/diversity_test.py`
Checks PGN diversity in `data/PGN`.
Reports:
- opening diversity (first N plies)
- results distribution
- game length stats

### `src/evaluation/stockfish_eval.py`
Value-head calibration against Stockfish on sampled positions.
- samples FENs from CSV `GM_GAMES_PATH`
- evaluates Stockfish score vs NN value
- reports MAE/MSE/correlation + sample FENs

---

## Logging

### `src/logging/discord_webhooks.py`
Minimal Discord webhook client + embed helpers.
- `send_webhook(url, payload)`
- `promotion_embed(...)`
- `rating_embed(...)`

### `hub.py`
Interactive CLI / orchestration.
- Self-play only
- Train RL only / cycle
- Evaluate & promote
- Run RL loop (self-play → train_rl → promote)
- Passes loop context into promotion logging (iteration/max, sims, etc.)

---

## Data preprocessing

### `preprocess_rust.py`
Build SL shards from a CSV using a Rust helper module to extract SAN moves fast.
- constants: `SAMPLE_RATE`, `SHARD_SIZE`
- `preprocess()`: reads `GM_GAMES_PATH`, writes `shard_*.pt` to `SHARD_DIR`

SL shard format:
- `boards`: `(N, 18, 8, 8)` float32
- `policies`: `(N, 4672)` float32 one-hot
- `values`: `(N, 1)` float32 in `[-1, 1]` (white outcome)
