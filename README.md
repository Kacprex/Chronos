# Chronos (Phase 1) — Supervised Learning (SL) + Two Streamlit Apps

This repository is **Phase 1 implemented end-to-end**:
- **Input:** positions as **FEN** lines (`raw_fens.txt`)
- **Labels:** Stockfish evals at fixed depth (parallel workers)
- **Dataset:** binary `dataset.bin` (18-plane encoding + label)
- **Training:** PyTorch SL training → exports `nn_sl.bin`
- **UI:** two Streamlit hubs
  - **Local Control Panel** (start jobs, view logs)
  - **Public Dashboard** (read-only metrics)

> Notes:
> - This phase intentionally avoids full PGN/SAN parsing (too heavy for Phase 1). You feed Chronos a **FEN list**.
> - Everything large is stored under `E:\chronos` (or another drive via `CHRONOS_DATA` env var).
> - C++ is included with CMake as the project base (encoding spec + smoke tool). Training/labeling are in Python.

---

## 0) Prerequisites

### Software
- Windows 10/11
- Python 3.10+ (recommended 3.11)
- CMake 3.20+
- A Stockfish binary (Windows): `stockfish.exe`

### Set the data root
Chronos uses a single environment variable:

```powershell
setx CHRONOS_DATA E:\chronos
```

Open a new PowerShell after setting it.

Default if not set: `E:\chronos`

### Put Stockfish here
Copy Stockfish to:

```
E:\chronos\private\bin\stockfish.exe
```

(or any path, but then pass `--stockfish`).

---

## 1) Data layout on your fast drive

Chronos creates this structure:

```
E:\chronos\
  private\
    sl\
      raw_fens.txt
      labeled.jsonl.gz
      dataset.bin
    models\
      nn_sl.bin
    logs\
      label_sf.log
      build_dataset.log
      train_sl.log
    control\
      status.json
      jobs.json
  public\
    metrics\
      sl_loss.csv
      sl_val_loss.csv
      sl_throughput.csv
    status.json
    latest_model.txt
    sample_fens.txt
```

**Security model:**
- `private\` is full access (local only)
- `public\` is safe to expose (read-only dashboard reads only this)

---

## 2) Quickstart (Phase 1)

### A) Create a Python venv and install deps

```powershell
cd C:\dev\chronos_phase1
python -m venv .venv
.\.venv\Scripts\activate
pip install -r python\requirements.txt
```

### B) Provide FENs

Create:

```
E:\chronos\private\sl\raw_fens.txt
```

with **one FEN per line** (examples in `E:\chronos\public\sample_fens.txt` after first run).

### C) Label with Stockfish (parallel)

```powershell
python python\phase1\label_sf.py `
  --in_fens  E:\chronos\private\sl\raw_fens.txt `
  --out_jsonl_gz E:\chronos\private\sl\labeled.jsonl.gz `
  --depth 10 `
  --workers 8
```

### D) Build dataset (encode FENs → 18 planes + normalized labels)

```powershell
python python\phase1\build_dataset.py `
  --in_labeled_jsonl_gz E:\chronos\private\sl\labeled.jsonl.gz `
  --out_dataset_bin E:\chronos\private\sl\dataset.bin `
  --label_scale_cp 600
```

### E) Train SL

```powershell
python python\phase1\train_sl.py `
  --dataset_bin E:\chronos\private\sl\dataset.bin `
  --out_model_bin E:\chronos\private\models\nn_sl.bin `
  --epochs 3 `
  --batch_size 4096 `
  --lr 3e-4
```

After training:
- `E:\chronos\private\models\nn_sl.bin` is produced
- public metrics update under `E:\chronos\public\metrics\`

---

## 3) Streamlit hubs

### Local control panel (PRIVATE)
Starts/monitors labeling/build/training, reads private logs.

```powershell
streamlit run python\streamlit\local_control\app.py --server.port 8501
```

### Public dashboard (READ-ONLY)
Shows only metrics from `E:\chronos\public\...`

```powershell
streamlit run python\streamlit\public_dashboard\app.py --server.port 8601 --server.address 0.0.0.0
```

**Important:** keep the public dashboard read-only and do not mount `private\` there.

---

## 4) Encoding spec (Phase 1)

Input vector is **18 × 8 × 8 = 1152 floats**:

- 12 planes: pieces (P,N,B,R,Q,K for white then black)
- 1 plane: side to move (all 1.0 if white to move else 0.0)
- 4 planes: castling rights (WK, WQ, BK, BQ)
- 1 plane: halfmove clock normalized to [0,1] by `min(halfmove,100)/100`

**No en-passant plane** in Phase 1.

Label:
- Stockfish `cp` (centipawns) mapped to [-1,1] by:
  - `y = clamp(cp / label_scale_cp, -1, 1)`

---

## 5) C++ (CMake) — included base

Phase 1 focuses on data/SL, but we include C++ scaffolding with:
- encoding spec header
- `chronos_encode_fen` smoke tool

Build:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\chronos_encode_fen.exe "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

---

## 6) What’s implemented vs deferred

Implemented in Phase 1:
- Stockfish parallel labeling (multiprocessing)
- Dataset builder (FEN → encoding → dataset.bin)
- SL trainer (PyTorch MLP)
- Export of weights to `nn_sl.bin`
- Local control Streamlit
- Public read-only Streamlit dashboard
- Private/public data separation

Deferred (Phase 2+):
- Self-play RL
- Model-vs-model promotion ladder
- Runtime batched GPU inference server
- Full PGN SAN parsing (recommended: external pre-processing → FEN list)

---

## License
MIT (see `LICENSE`).
