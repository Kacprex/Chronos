# Phases 8–10 — Policy head + MCTS selfplay + hybrid training + engine priors

## Phase 8 — Action space + policy head
- Adds AlphaZero-style move indexing: **8x8x73 = 4672**
  - C++: `engine/include/chronos/nn/move_index.h`
  - Python: `nn/move_index.py`
- Adds hybrid model with policy head:
  - `nn/models/chronos_hybrid.py`
- ONNX export now supports:
  - scalar-only output `[B,4]`
  - hybrid output `[B, 4+4672]` = scalars + policy logits

## Phase 9 — MCTS selfplay (scaffold)
- `python -m rl.selfplay.run_mcts_selfplay --ckpt <hybrid_latest_model.pt>`
- Output:
  - `E:/chronos/shards/az/<run_id>/az_selfplay.jsonl`

## Phase 10 — Use policy priors in engine hybrid selection
- Engine now uses **policy priors** (softmax over candidate move logits) to slightly bias selection among near-best lines.
  - This is *not* full NN-guided search yet — it’s a safe, cheap step that already changes “style”.

## Quick commands
### A) Make policy targets from Stockfish (fast)
1) Selfplay to create raw:
```powershell
python -m rl.selfplay.run_selfplay --engine .\build\Release\chronos_engine.exe --games 50 --movetime-ms 100
```
2) Label with MultiPV:
```powershell
python -m rl.analyze.label_stockfish --raw E:\chronos\shards\rl\<run_id>\selfplay_raw.jsonl --stockfish E:\path\stockfish.exe --depth 12 --multipv 5
```
3) Train hybrid:
```powershell
python -m rl.training.train_hybrid_policy --data E:\chronos\shards\rl\<run_id>\labeled.jsonl --epochs 3
```
4) Export ONNX:
```powershell
python nn/export/export_onnx.py --ckpt E:\chronos\runs\<train_run>\hybrid_latest_model.pt --out E:\chronos\models\candidate_hybrid.onnx
```

### B) MCTS selfplay (slower, more RL-like)
```powershell
python -m rl.selfplay.run_mcts_selfplay --ckpt E:\chronos\runs\<train_run>\hybrid_latest_model.pt --games 10 --sims 200
```
Then train on `az_selfplay.jsonl` with `train_hybrid_policy`.
