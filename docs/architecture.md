# Architecture (Phases 0–1)

- `engine/` : C++ UCI chess engine core (movegen, search, eval, TT)
- `nn/`     : PyTorch training + export (Phase 5+)
- `hub/`    : Streamlit dashboard (Phase 8)
- `notifier/`: Discord webhook notifier (Phase 8)
- `tools/`  : Rust/utility tools (PGN → tensors, shard tools) (Phase 5+)

All persistent artifacts are intended to live under `E:/chronos` by default (see `configs/engine.yaml`).
