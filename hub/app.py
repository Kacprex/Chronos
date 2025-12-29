from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

DEFAULT_ROOT = Path(os.environ.get("CHRONOS_DATA_ROOT", r"E:/chronos"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Chronos Hub", layout="wide")
st.title("🕰️ Chronos Hub — Phases 6–7")
st.caption("Extract shards, run pretrain, export ONNX, inspect logs/curves, and optionally ping Discord.")


def events_path(root: Path) -> Path:
    return root / "logs" / "events.jsonl"


def send_discord(webhook: str, content: str) -> Tuple[bool, str]:
    try:
        r = requests.post(webhook, json={"content": content}, timeout=10)
        if 200 <= r.status_code < 300:
            return True, "sent"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


def run_cmd(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out


def read_last_events(path: Path, n: int = 200) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
    ev = []
    for ln in lines:
        try:
            ev.append(json.loads(ln))
        except Exception:
            pass
    return ev


st.sidebar.header("Settings")
root = Path(st.sidebar.text_input("Data root", str(DEFAULT_ROOT))).expanduser()
root.mkdir(parents=True, exist_ok=True)

webhook = st.sidebar.text_input(
    "Discord webhook (optional)",
    value=os.environ.get("CHRONOS_DISCORD_WEBHOOK", ""),
    type="password",
)
ping_on_finish = st.sidebar.toggle("Ping Discord on job finish", value=bool(webhook))

if st.sidebar.button("Send test ping") and webhook:
    ok, msg = send_discord(webhook, "🕰️ Chronos Hub test ping.")
    st.sidebar.success("Ping sent." if ok else f"Ping failed: {msg}")


tabs = st.tabs(
    [
        "Overview",
        "Logs",
        "Extract (PGN → shards)",
        "Pretrain (shards → .pt)",
        "Export (pt → onnx)",
        "Engine NN (Phase 6)",
        "RL (Selfplay → Label → Train → Promote)",
        "AZ (MCTS Selfplay → Train Hybrid)",
    ]
)

with tabs[0]:
    st.subheader("Folders")
    st.write("Runs:", root / "runs")
    st.write("Shards:", root / "shards")
    st.write("Models:", root / "models")
    st.write("Logs:", root / "logs")
    st.write("Events:", events_path(root))

    st.subheader("Latest events")
    ev = read_last_events(events_path(root), n=200)
    if not ev:
        st.info("No events found yet.")
    else:
        st.dataframe(pd.DataFrame(ev[-50:]).tail(50), use_container_width=True)

with tabs[1]:
    st.subheader("events.jsonl tail")
    n = st.slider("Lines", 50, 2000, 300, step=50)
    ev = read_last_events(events_path(root), n=n)
    if ev:
        df = pd.DataFrame(ev)
        if "type" in df.columns:
            typ = st.multiselect("Filter by type", sorted(df["type"].dropna().unique().tolist()))
            if typ:
                df = df[df["type"].isin(typ)]
        st.dataframe(df.tail(200), use_container_width=True)
        st.download_button(
            "Download tail as JSON",
            data=json.dumps(ev, ensure_ascii=False, indent=2),
            file_name="events_tail.json",
            mime="application/json",
        )
    else:
        st.info("No events yet.")

with tabs[2]:
    st.subheader("Extract shards from PGN")
    st.write("Default output:", root / "shards" / "sl" / "<dataset>" / "shards.jsonl")

    pgn = st.text_input("PGN path", value="")
    dataset = st.text_input("Dataset name (folder)", value="")
    min_moves = st.number_input("Min moves", 10, 200, 40)
    start_move = st.number_input("Start move", 1, 60, 12)
    end_move = st.number_input("End move", 1, 200, 60)
    max_games = st.number_input("Max games (0 = all)", 0, 1_000_000, 0)

    if st.button("Run extraction"):
        if not pgn:
            st.error("Provide a PGN path.")
        else:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "extract_shards.py"),
                "--pgn",
                pgn,
                "--min-moves",
                str(min_moves),
                "--start-move",
                str(start_move),
                "--end-move",
                str(end_move),
                "--max-games",
                str(max_games),
            ]
            if dataset:
                cmd += ["--name", dataset]

            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)

            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=250)

            if rc == 0:
                st.success("Extraction complete.")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"✅ Chronos: shard extraction finished. PGN: `{pgn}`")
            else:
                st.error("Extraction failed (see output).")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"❌ Chronos: shard extraction FAILED. PGN: `{pgn}`")

with tabs[3]:
    st.subheader("Pretrain multi-head CNN (Phase 5 scaffold)")
    shards = st.text_input("Shards JSONL path", value="")
    run_id = st.text_input("Run id (optional)", value="")
    epochs = st.number_input("Epochs", 1, 100, 3)
    batch = st.number_input("Batch size", 8, 4096, 256, step=8)
    lr = st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, format="%.6f")

    if st.button("Run pretrain"):
        if not shards:
            st.error("Provide shards JSONL path.")
        else:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "nn" / "training" / "pretrain.py"),
                "--shards",
                shards,
                "--epochs",
                str(int(epochs)),
                "--batch",
                str(int(batch)),
                "--lr",
                str(float(lr)),
            ]
            if run_id:
                cmd += ["--run", run_id]

            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)

            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=250)

            if rc == 0:
                st.success("Pretrain complete. Check E:/chronos/runs/<run_id>/metrics.csv and latest_model.pt")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"✅ Chronos: pretrain finished. Run: `{run_id or 'auto'}`")
            else:
                st.error("Pretrain failed (see output).")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"❌ Chronos: pretrain FAILED. Run: `{run_id or 'auto'}`")

    st.subheader("Plot metrics.csv")
    metrics_path = st.text_input("metrics.csv path", value="")
    if st.button("Load metrics"):
        mp = Path(metrics_path)
        if mp.exists():
            df = pd.read_csv(mp)
            st.dataframe(df, use_container_width=True)
            if "epoch" in df.columns:
                idx = df.set_index("epoch")
                cols = [c for c in ["loss", "loss_value", "loss_pressure", "loss_volatility", "loss_complexity"] if c in idx.columns]
                if cols:
                    st.line_chart(idx[cols])
        else:
            st.error("metrics.csv not found.")

with tabs[4]:
    st.subheader("Export ONNX model")
    ckpt = st.text_input("Checkpoint (.pt) path", value="")
    outp = st.text_input("Output ONNX path (optional)", value="")

    if st.button("Export"):
        if not ckpt:
            st.error("Provide checkpoint path.")
        else:
            cmd = [sys.executable, str(PROJECT_ROOT / "nn" / "export" / "export_onnx.py"), "--ckpt", ckpt]
            if outp:
                cmd += ["--out", outp]

            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)

            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=200)

            if rc == 0:
                st.success("Export complete.")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"✅ Chronos: ONNX export finished. Ckpt: `{ckpt}`")
            else:
                st.error("Export failed.")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"❌ Chronos: ONNX export FAILED. Ckpt: `{ckpt}`")

with tabs[5]:
    st.subheader("Engine neural inference (Phase 6)")
    st.write("Chronos supports optional ONNX Runtime inference via CMake (Windows example):")
    st.code(
        "cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release ^\n"
        "  -DCHRONOS_WITH_ONNX=ON ^\n"
        "  -DCHRONOS_ONNX_ROOT=C:\\path\\to\\onnxruntime-win-x64-...\n\n"
        "cmake --build build --config Release"
    )

    st.write("UCI options:")
    st.code(
        "setoption name UseNN value true\n"
        "setoption name NNModel value E:/chronos/models/chronos.onnx\n"
        "setoption name NNIntraThreads value 1\n"
        "setoption name NNInterThreads value 1\n"
        "setoption name NNPreferCuda value false"
    )

    st.info("If ONNX is not compiled in, logs will include nn_error explaining why UseNN is inactive.")



with tabs[6]:
    st.subheader("RL training loop (Phase 7 RL)")
    st.write("Pipeline: **Selfplay** → **Stockfish labeling** → **Train heads** → **Export ONNX** → (optional) **Promotion games**")

    st.markdown("### 1) Selfplay")
    engine_path = st.text_input("Chronos engine path", value="")
    sp_run = st.text_input("Selfplay run id (optional)", value="")
    sp_games = st.number_input("Games", 1, 100000, 20)
    sp_movetime = st.number_input("Move time (ms)", 10, 5000, 100, step=10)
    sp_depth = st.number_input("Depth (0 = use movetime)", 0, 50, 0)
    sp_mode = st.selectbox("Mode", ["blitz", "classic"], index=0)

    colA, colB = st.columns(2)
    with colA:
        sp_accept = st.number_input("Accept worse (cp)", 0, 500, 40)
    with colB:
        sp_topk = st.number_input("TopK", 1, 20, 8)

    if st.button("Run selfplay"):
        if not engine_path:
            st.error("Provide chronos engine path.")
        else:
            cmd = [
                sys.executable, "-m", "rl.selfplay.run_selfplay",
                "--engine", engine_path,
                "--games", str(int(sp_games)),
                "--movetime-ms", str(int(sp_movetime)),
                "--mode", sp_mode,
                "--accept-worse-cp", str(int(sp_accept)),
                "--topk", str(int(sp_topk)),
            ]
            if int(sp_depth) > 0:
                cmd += ["--depth", str(int(sp_depth))]
            if sp_run:
                cmd += ["--run", sp_run]

            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)

            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=220)
            if rc == 0:
                st.success("Selfplay complete.")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"✅ Chronos RL: selfplay finished. Run: `{sp_run or 'auto'}`")
            else:
                st.error("Selfplay failed.")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"❌ Chronos RL: selfplay FAILED. Run: `{sp_run or 'auto'}`")

    st.markdown("### 2) Label with Stockfish")
    raw_jsonl = st.text_input("selfplay_raw.jsonl path", value="")
    sf_path = st.text_input("Stockfish path", value="")
    lab_depth = st.number_input("SF depth", 1, 40, 12)
    lab_mistake = st.number_input("Mistake threshold (cp)", 10, 500, 70, step=10)
    lab_horizon = st.number_input("Horizon (plies)", 1, 20, 4)

    if st.button("Run labeling"):
        if not raw_jsonl or not sf_path:
            st.error("Provide raw jsonl and stockfish path.")
        else:
            cmd = [
                sys.executable, "-m", "rl.analyze.label_stockfish",
                "--raw", raw_jsonl,
                "--stockfish", sf_path,
                "--depth", str(int(lab_depth)),
                "--mistake-cp", str(int(lab_mistake)),
                "--horizon", str(int(lab_horizon)),
            ]
            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)
            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=220)
            if rc == 0:
                st.success("Labeling complete (labeled.jsonl next to raw).")
                if webhook and ping_on_finish:
                    send_discord(webhook, "✅ Chronos RL: Stockfish labeling finished.")
            else:
                st.error("Labeling failed.")
                if webhook and ping_on_finish:
                    send_discord(webhook, "❌ Chronos RL: Stockfish labeling FAILED.")

    st.markdown("### 3) Train heads on labeled dataset")
    labeled_path = st.text_input("labeled.jsonl path", value="")
    tr_run = st.text_input("Train run id (optional)", value="")
    tr_epochs = st.number_input("Epochs", 1, 200, 3)
    tr_batch = st.number_input("Batch", 8, 4096, 256, step=8)
    tr_lr = st.number_input("LR", 1e-6, 1e-1, 1e-3, format="%.6f")

    if st.button("Run RL training"):
        if not labeled_path:
            st.error("Provide labeled.jsonl path.")
        else:
            cmd = [
                sys.executable, "-m", "rl.training.train_rl_heads",
                "--labeled", labeled_path,
                "--epochs", str(int(tr_epochs)),
                "--batch", str(int(tr_batch)),
                "--lr", str(float(tr_lr)),
            ]
            if tr_run:
                cmd += ["--run", tr_run]
            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)
            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=220)
            if rc == 0:
                st.success("RL training complete (see runs/<run_id>/rl_metrics.csv).")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"✅ Chronos RL: training finished. Run: `{tr_run or 'auto'}`")
            else:
                st.error("RL training failed.")
                if webhook and ping_on_finish:
                    send_discord(webhook, f"❌ Chronos RL: training FAILED. Run: `{tr_run or 'auto'}`")

    st.markdown("### 4) Promotion games (optional, requires Phase 6 ONNX build)")
    promo_engine = st.text_input("Engine path for promo", value="")
    promo_cand = st.text_input("Candidate ONNX", value="")
    promo_best = st.text_input("Best ONNX", value="")
    promo_games = st.number_input("Promo games", 2, 200, 20)
    promo_ms = st.number_input("Promo movetime (ms)", 10, 5000, 100, step=10)
    promo_thr = st.number_input("Promotion threshold", 0.50, 0.90, 0.55)

    if st.button("Run promotion match"):
        if not promo_engine or not promo_cand or not promo_best:
            st.error("Provide engine + candidate + best ONNX paths.")
        else:
            cmd = [
                sys.executable, "-m", "rl.eval.promo",
                "--engine", promo_engine,
                "--candidate", promo_cand,
                "--best", promo_best,
                "--games", str(int(promo_games)),
                "--movetime-ms", str(int(promo_ms)),
                "--threshold", str(float(promo_thr)),
            ]
            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)
            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=220)
            if rc == 0:
                st.success("Promotion match complete (see output).")
                if webhook and ping_on_finish:
                    send_discord(webhook, "✅ Chronos RL: promotion match finished.")
            else:
                st.error("Promotion match failed.")
                if webhook and ping_on_finish:
                    send_discord(webhook, "❌ Chronos RL: promotion match FAILED.")




with tabs[7]:
    st.subheader("AZ-style loop (Phase 8–10)")
    st.write("Two ways to get policy targets:")
    st.markdown("- **Stockfish MultiPV**: label `labeled.jsonl` with `--multipv` (fast, supervised-ish)")
    st.markdown("- **MCTS selfplay**: generate `az_selfplay.jsonl` with visit-count policies (slower, more RL-like)")

    st.markdown("### A) Label with Stockfish MultiPV (policy targets)")
    raw_jsonl2 = st.text_input("Raw selfplay JSONL (from RL selfplay)", value="", key="raw_jsonl2")
    sf2 = st.text_input("Stockfish path (for MultiPV)", value="", key="sf2")
    mpv = st.number_input("MultiPV", 0, 20, 5, key="mpv")
    temp = st.number_input("Policy temperature (cp)", 10.0, 1000.0, 150.0, key="temp")
    dep = st.number_input("Depth", 1, 40, 12, key="dep")

    if st.button("Run MultiPV labeling"):
        if not raw_jsonl2 or not sf2:
            st.error("Provide raw jsonl and stockfish.")
        else:
            cmd = [
                sys.executable, "-m", "rl.analyze.label_stockfish",
                "--raw", raw_jsonl2,
                "--stockfish", sf2,
                "--depth", str(int(dep)),
                "--multipv", str(int(mpv)),
                "--policy-temp", str(float(temp)),
            ]
            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)
            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=220)

    st.markdown("### B) MCTS selfplay (needs hybrid checkpoint)")
    hy_ckpt = st.text_input("Hybrid checkpoint (.pt, kind=hybrid_policy)", value="", key="hy_ckpt")
    az_run = st.text_input("AZ run id (optional)", value="", key="az_run")
    az_games = st.number_input("Games", 1, 100000, 5, key="az_games")
    az_sims = st.number_input("Sims per move", 10, 5000, 200, key="az_sims")
    az_temp = st.number_input("Move temperature", 0.0, 5.0, 1.0, key="az_temp")

    if st.button("Run AZ selfplay"):
        if not hy_ckpt:
            st.error("Provide hybrid checkpoint.")
        else:
            cmd = [
                sys.executable, "-m", "rl.selfplay.run_mcts_selfplay",
                "--ckpt", hy_ckpt,
                "--games", str(int(az_games)),
                "--sims", str(int(az_sims)),
                "--temp", str(float(az_temp)),
            ]
            if az_run:
                cmd += ["--run", az_run]
            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)
            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=220)

    st.markdown("### C) Train hybrid (policy + scalar heads)")
    data_jsonl = st.text_input("Training JSONL (labeled.jsonl with policy OR az_selfplay.jsonl)", value="", key="data_jsonl")
    tr_run2 = st.text_input("Train run id (optional)", value="", key="tr_run2")
    ep2 = st.number_input("Epochs", 1, 100, 3, key="ep2")
    bs2 = st.number_input("Batch", 8, 2048, 128, step=8, key="bs2")
    lr2 = st.number_input("LR", 1e-6, 1e-1, 1e-3, format="%.6f", key="lr2")

    if st.button("Train hybrid"):
        if not data_jsonl:
            st.error("Provide training JSONL.")
        else:
            cmd = [
                sys.executable, "-m", "rl.training.train_hybrid_policy",
                "--data", data_jsonl,
                "--epochs", str(int(ep2)),
                "--batch", str(int(bs2)),
                "--lr", str(float(lr2)),
            ]
            if tr_run2:
                cmd += ["--run", tr_run2]
            env = os.environ.copy()
            env["CHRONOS_DATA_ROOT"] = str(root)
            st.code(" ".join(cmd))
            rc, outtxt = run_cmd(cmd, cwd=PROJECT_ROOT, env=env)
            st.text_area("Output", outtxt, height=220)
