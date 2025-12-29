# Chronos (Phases 0–1)

This zip contains **Phase 0 (project skeleton + config/logging conventions)**, **Phase 1 (a working C++ UCI chess engine MVP)**, **Phase 2 (eval breakdown logging)**, **Phase 3 (neural inference hook stub / optional ONNX)**, **Phase 4 (hybrid arbitration logic)** and **Phase 5 (GM pretraining scaffold)**.

## What you get (Phase 0)
- Clean repo layout (engine / nn / hub / notifier / tools / scripts / configs / tests / docs)
- Central config in `configs/engine.yaml`
- A single source of truth for storage paths:
  - **All shards are intended to live under `E:/chronos`** (configurable but defaulted accordingly)

## What you get (Phase 1)
- `chronos_engine` (C++17) with:
  - Legal move generation (incl. castling + en passant + promotions)
  - Make/unmake with undo stack
  - Alpha-beta search + quiescence (captures)
  - Transposition table
  - Basic evaluation (material + PST)
  - UCI interface (`uci`, `isready`, `position`, `go`, `stop`, `quit`)

> This is an MVP meant to be correct and hackable. Performance is good enough for iteration and later optimization.

---

## Build (Windows / Visual Studio 2022)

### Option A: CMake + Visual Studio generator
From the repo root:

```powershell
mkdir build
cmake -S engine -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The executable will be in:
`build/Release/chronos_engine.exe`

### Option B: CMake + Ninja (recommended if you have Ninja)
```powershell
cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Quick UCI test
Run `chronos_engine.exe` and type:

```
uci
isready
position startpos
go depth 5
```

---

## Storage paths (IMPORTANT)
Default storage root is **`E:/chronos`** via `configs/engine.yaml`.

Nothing in Phase 0–1 generates shards yet, but the path conventions and helpers are already in place:
- `E:/chronos/shards/sl/...`
- `E:/chronos/shards/rl/...`

---

## Next split (later)
When you proceed, you can split into:
- `sl/` (supervised learning / GM pretrain)
- `rl/` (self-play + shard selection + promotion)

and keep all shards under `E:/chronos/shards`.



## Phases 6–7 (added)
- **Phase 6:** optional ONNX Runtime inference in the C++ engine.
- **Phase 7:** upgraded Streamlit Hub + optional Discord webhook watcher.


## Phase 7 (RL training loop)
See `docs/phase_7_rl.md` and the `rl/` package for selfplay → Stockfish labeling → training → promotion.


## Phases 8–10
- **Phase 8:** AlphaZero-style move indexing (4672) + policy head + ONNX export `[4+4672]`.
- **Phase 9:** MCTS selfplay scaffold producing visit-count policies.
- **Phase 10:** Engine hybrid selection uses policy priors for style bias.
See `docs/phases_8_10.md`.
