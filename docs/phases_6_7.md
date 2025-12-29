# Phases 6–7

## Phase 6 — ONNX Runtime inference in the C++ engine (optional build)
Chronos can now load and run an ONNX model **if** you build with:
- `-DCHRONOS_WITH_ONNX=ON`
- `-DCHRONOS_ONNX_ROOT=...` pointing at an ONNX Runtime distribution folder

Expected tensors:
- input name: `input`
- output name: `output`
- input: `[1, 25, 8, 8]` float32
- output: `[1, 4]` float32 = `[value, pressure, volatility, complexity]`

New UCI options:
- `NNIntraThreads`, `NNInterThreads`, `NNPreferCuda` (placeholder until you link CUDA EP)

If ONNX is not compiled in, `UseNN` will not become **active** and logs include `nn_error`.

## Phase 7 — Hub + Discord
- `hub/app.py` upgraded to:
  - run PGN → shards extraction
  - run shards → pretrain
  - run checkpoint → ONNX export
  - inspect `events.jsonl` and plot metrics

Discord watcher:
- `python notifier/watch_events.py --webhook <url> --root E:/chronos`


## Phase 7 (RL)
See `docs/phase_7_rl.md` and the `rl/` package for the selfplay→label→train loop.
