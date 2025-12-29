# Phases 4–5

## Phase 4: Hybrid arbitration ("Chronos style")
Chronos can accept slightly worse evaluation to pick a move that:
- increases opponent decision complexity (proxy),
- avoids early simplification,
- and later (with a real NN) maximizes learned pressure.

UCI options:
- `Hybrid` (true/false)
- `Mode` ("classic" or "blitz")
- `AcceptWorseCp` (centipawns margin from best classical)
- `TopK` (how many root candidates to consider)

The engine logs candidates + chosen move to:
`E:/chronos/logs/events.jsonl`

## Phase 5: GM pretraining scaffold
Tools:
- `tools/extract_shards.py` reads a PGN and writes shard records to:
  `E:/chronos/shards/sl/<dataset>/shards.jsonl`

Neural:
- `nn/models/chronos_cnn.py` multi-head CNN
- `nn/training/pretrain.py` trains on shards JSONL and writes to:
  `E:/chronos/runs/<run_id>/...` and emits events to `E:/chronos/logs/events.jsonl`
- `nn/export/export_onnx.py` exports ONNX to:
  `E:/chronos/models/chronos.onnx`

Note: Phase 5 targets are bootstrapped proxies. Later phases will replace them with
engine/Stockfish-assisted labels + shard scoring.
