# Phases 2–3

## Phase 2: Eval breakdown logging
- `evaluate_breakdown()` splits evaluation into components (currently `material` + `pst` + placeholders).
- After each `go`, the engine appends a JSON object to:

`E:/chronos/logs/events.jsonl` (default)

with:
- best move
- search depth/nodes/score
- `eval_before` and (if legal) `eval_after`

## Phase 3: Neural hook
- Position encoder produces 25 planes (12 piece planes + stm + castling + ep-file planes).
- `NNEvaluator` compiles without ONNX Runtime:
  - if `UseNN` is enabled but ONNX isn't built in, outputs are zeroed (stub).
- Future: enable ONNX by building with `-DCHRONOS_WITH_ONNX=ON` and adding ONNX Runtime include/lib.

UCI options:
- `setoption name UseNN value true`
- `setoption name NNModel value <path_to_model.onnx>`
- `setoption name Log value true`
- `setoption name LogPath value E:/chronos/logs/events.jsonl`
