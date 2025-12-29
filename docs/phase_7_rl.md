# Phase 7 (RL) — selfplay → Stockfish labeling → train → export → promote

This phase adds a practical RL-style loop focused on **punishing impatience**.

## Files written under E:/chronos
- Selfplay raw:
  - `E:/chronos/shards/rl/<run_id>/selfplay_raw.jsonl`
  - `E:/chronos/shards/rl/<run_id>/selfplay.pgn`
  - engine logs per run:
    - `E:/chronos/shards/rl/<run_id>/engine_events.jsonl`
- Labeled dataset:
  - `E:/chronos/shards/rl/<run_id>/labeled.jsonl`
- Training outputs:
  - `E:/chronos/runs/<run_id>/rl_latest_model.pt`
  - `E:/chronos/runs/<run_id>/rl_metrics.csv`
- Promotion (if enabled):
  - `E:/chronos/models/best.onnx`
  - `E:/chronos/logs/promotions.txt`

## Step 1 — Selfplay
```powershell
python -m rl.selfplay.run_selfplay --engine .\build\Release\chronos_engine.exe --games 50 --movetime-ms 100 --mode blitz
```

## Step 2 — Label with Stockfish
```powershell
python -m rl.analyze.label_stockfish --raw E:\chronos\shards\rl\<run_id>\selfplay_raw.jsonl --stockfish E:\path\stockfish.exe --depth 12 --mistake-cp 70 --horizon 4
```

Labels produce targets:
- **value**: tanh(stockfish_cp/600) from side-to-move POV
- **pressure**: opponent makes a "mistake" within next horizon plies
- **volatility**: max eval swing within horizon (normalized)
- **complexity**: legal_moves/40 (proxy)

## Step 3 — Train heads on labeled RL dataset
```powershell
python -m rl.training.train_rl_heads --labeled E:\chronos\shards\rl\<run_id>\labeled.jsonl --epochs 5 --batch 256 --lr 1e-3
```

## Step 4 — Export ONNX
Use existing exporter:
```powershell
python nn/export/export_onnx.py --ckpt E:\chronos\runs\<train_run>\rl_latest_model.pt --out E:\chronos\models\candidate.onnx
```

## Step 5 — Promotion matches (optional)
Requires engine built with ONNX runtime (Phase 6).
```powershell
python -m rl.eval.promo --engine .\build\Release\chronos_engine.exe --candidate E:\chronos\models\candidate.onnx --best E:\chronos\models\best.onnx --games 40 --movetime-ms 100 --threshold 0.55
```
