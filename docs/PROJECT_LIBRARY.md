# Chronos project library

A “library index” of the codebase: **what each file is for, what it exports, and how it’s used**.

---

## High-level architecture

1. **Encode** a position → 18×8×8 planes (`src/nn/encoding.py`)
2. **Neural net** → policy logits (4672) + value (-1..1) (`src/nn/network.py`)
3. **MCTS** refines policy (`src/mcts/mcts.py`)
4. **Training**
   - SL: imitate human next-move + game outcome
   - RL: learn from MCTS policy targets + self-play outcome
5. **Promotion**: latest vs best (`src/evaluation/promotion.py`)

---

## File-by-file

### `src/config.py`
Central configuration: paths and output dirs.
- `GM_GAMES_PATH`, `SHARD_DIR`, `RL_BUFFER_DIR`
- `PHASE1_MODEL_PATH`, `BEST_MODEL_PATH`, `LATEST_MODEL_PATH`
- `ENGINE_PATH`, PGN output paths (`AIVSAI`, `SFVSAI`)

---

### `src/nn/encoding.py`
Board encoding + AlphaZero-style move indexing.
- `MOVE_SPACE = 4672`
- `encode_board(board) -> (18,8,8)` float32 planes
- `move_to_index(move) -> int|None`
- `index_to_move(index, board) -> chess.Move` (scans legal moves)

---

### `src/nn/network.py`
Policy+value CNN.
- `ResidualBlock`
- `ChessNet(channels=128, num_res_blocks=5)`
  - output policy logits: `(B,4672)`
  - output value: `(B,1)` in `[-1,1]`

---

### `src/mcts/mcts.py`
The active MCTS.
- `Node` dataclass (board, prior, visits, value_sum, children)
- `MCTS.run(board, move_number, add_noise=True) -> (moves, probs)`
  - root expansion + optional Dirichlet noise
  - `simulations` playouts
  - returns probability distribution derived from child visits

---

### `src/selfplay/opening_book.py`
Small curated opening set.
- `OPENINGS`
- `play_random_opening(board, max_plies=6) -> int`

---

### `src/selfplay/encode_game.py`
PGN writing utilities.
- `GameRecord` (moves_uci, result, headers; can append to file safely)
- `build_game_record_from_moves(moves, result, ...) -> GameRecord`

---

### `src/selfplay/self_play_worker.py`
Self-play + RL shard writer.
- `play_single_game(...) -> samples`
- `self_play(num_games, simulations, shard_size)`
  - loads `BEST_MODEL_PATH`
  - writes `rl_shard_*.pt` to `RL_BUFFER_DIR`

⚠ Known issue: indentation/import artifact currently present.

---

### `src/training/train_supervised.py`
Supervised training from SL shards.
- resume checkpoint: `models/checkpoints/phase1_resume.pt`
- key functions: `get_shard_paths`, `load_resume_state`, `save_checkpoint`, `train`

Loss:
- policy: `CrossEntropyLoss(policy_logits, argmax(onehot_policy))`
- value: `MSELoss(value_pred, value_target)`

---

### `src/training/train_rl.py`
RL training from RL shards.
- resume checkpoint: `models/checkpoints/rl_resume.pt`
- key functions: `get_rl_shards`, `load_resume_state`, `save_checkpoint`, `train_rl`

Loss:
- policy: `KLDivLoss(log_softmax(logits), target_pi)`
- value: `MSELoss(value_pred, value_target)`

Output:
- writes `LATEST_MODEL_PATH`

---

### `src/evaluation/promotion.py`
Promotion check: latest vs best.
- `play_game(...) -> score`
- `evaluate_and_promote(num_games=50, threshold=0.55)`
  - alternates colors
  - promotes if average score ≥ threshold

---

### `src/evaluation/diversity_test.py`
Checks PGN diversity in `data/PGN`.
Reports:
- opening diversity (first N plies)
- results distribution
- game length stats

---

### `src/evaluation/stockfish_eval.py`
Experimental: value head vs Stockfish on sampled positions.

⚠ Status: inconsistent/unreliable (import/path mixups).

---

### `hub.py`
Interactive CLI.
- `play_ai_vs_ai(...)`
- `play_sf_vs_ai(...)`
- `main_menu()`

---

### `preprocess_rust.py`
Build SL shards from a CSV using a Rust helper module (`rust_pgn`) to extract SAN moves fast.
- constants: `SAMPLE_RATE`, `SHARD_SIZE`
- `preprocess()`: reads `GM_GAMES_PATH`, writes `shard_*.pt` to `SHARD_DIR`
