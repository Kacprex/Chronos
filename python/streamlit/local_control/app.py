from __future__ import annotations

import html as _html
import json
import os
import subprocess
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = REPO_ROOT / "data" / "logs"
STATUS_DIR = REPO_ROOT / "data" / "status"

# Prefer venv python if present; fallback to current interpreter
PY = (REPO_ROOT / ".venv" / "Scripts" / "python.exe")
if not PY.exists():
    PY = Path(os.environ.get("PYTHON", "")) if os.environ.get("PYTHON") else Path(subprocess.list2cmdline([os.sys.executable]))
    # If above produced a quoted string, just use sys.executable
    try:
        PY = Path(os.sys.executable)
    except Exception:
        PY = Path("python")

WRAPPER = REPO_ROOT / "python" / "streamlit" / "common" / "run_with_ts.py"

# -----------------------------
# Jobs
# -----------------------------
JOBS = {
    "build_dataset": {
        "title": "Build dataset.bin",
        "status_file": "build_dataset.json",
        "log_file": "build_dataset.log",
        "cmd": [
            str(PY),
            "-u",
            "python/phase1/build_dataset.py",
            "--out_bin",
            "data/datasets/dataset.bin",
        ],
    },
    "label_sf": {
        "title": "Label with Stockfish",
        "status_file": "label_sf.json",
        "log_file": "label_sf.log",
        "cmd": [
            str(PY),
            "-u",
            "python/phase1/label_sf.py",
            "--in_pgn",
            "data/pgn/input.pgn",
            "--out_labels",
            "data/labels/labels.jsonl",
            "--engine",
            "engine/stockfish.exe",
        ],
    },
    "train_sl": {
        "title": "Train SL model",
        "status_file": "train_sl.json",
        "log_file": "train_sl.log",
        "cmd": [
            str(PY),
            "-u",
            "python/phase1/train_sl.py",
            "--dataset_bin",
            "data/datasets/dataset.bin",
            "--out_model_bin",
            "data/models/nn_sl.bin",
            "--epochs",
            "3",
            "--batch_size",
            "4096",
            "--lr",
            "0.0003",
        ],
    },
}


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)


def read_status(name: str) -> dict:
    p = STATUS_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_logs() -> list[str]:
    ensure_dirs()
    logs = sorted([p.name for p in LOGS_DIR.glob("*.log")])
    return logs


@st.cache_data(show_spinner=False, ttl=1)
def read_log_tail(log_name: str, max_lines: int = 3000) -> str:
    p = LOGS_DIR / log_name
    if not p.exists():
        return ""
    try:
        with p.open("rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        return f"[error reading log: {e}]"


def render_log_box(text: str, height: int = 520) -> None:
    safe = _html.escape(text or "")
    components.html(
        f"""
<div style="height:{height}px; overflow-y:auto; border:1px solid rgba(255,255,255,0.15); border-radius:12px; padding:12px; background:#0e1117; color:#fafafa; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size:12px; line-height:1.35; white-space:pre;">
{safe}
</div>
""",
        height=height + 30,
        scrolling=True,
    )


def clear_logs(which: str | None = None) -> None:
    ensure_dirs()
    if which:
        paths = [LOGS_DIR / which]
    else:
        paths = list(LOGS_DIR.glob("*.log"))
    for p in paths:
        try:
            p.write_text("", encoding="utf-8")
        except Exception:
            pass
    # Make sure the UI picks up the change immediately.
    read_log_tail.clear()


def run_bg(job_key: str) -> None:
    ensure_dirs()
    job = JOBS[job_key]
    log_path = LOGS_DIR / job["log_file"]

    # Run via timestamping wrapper so *every* line has a timestamp.
    cmd = job["cmd"]
    wrapped_cmd = [str(PY), "-u", str(WRAPPER), "--log", str(log_path), "--"] + cmd

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)

    subprocess.Popen(
        wrapped_cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Chronos – Local Control", layout="wide")

st.title("Chronos – Local Control")
st.caption("Run Phase 1 scripts locally and monitor logs/status.")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Actions")
    for key, job in JOBS.items():
        cols = st.columns([2, 1])
        with cols[0]:
            st.write(job["title"])
        with cols[1]:
            if st.button("Run", key=f"run_{key}"):
                run_bg(key)
                st.success(f"Started: {job['title']}")

    st.divider()
    st.subheader("Statuses")
    for key, job in JOBS.items():
        s = read_status(job["status_file"])
        if not s:
            st.write(f"**{job['title']}** – (no status yet)")
            continue
        state = s.get("state", "?")
        msg = s.get("msg", "")
        p = s.get("progress", {})
        pct = p.get("pct")
        if isinstance(pct, (int, float)):
            st.write(f"**{job['title']}** – {state} ({pct:.1f}%) {msg}")
        else:
            st.write(f"**{job['title']}** – {state} {msg}")

with right:
    st.subheader("Logs")

    logs = list_logs()
    if not logs:
        st.info("No logs yet. Run a job on the left.")
    else:
        top = st.columns([3, 1, 1, 1])
        with top[0]:
            log_name = st.selectbox("Select log", logs, key="log_select_local")
        with top[1]:
            max_lines = st.number_input("Tail lines", min_value=200, max_value=20000, value=3000, step=200)
        with top[2]:
            refresh_s = st.selectbox("Auto-refresh", [0, 1, 2, 5, 10], index=1)
        with top[3]:
            if st.button("Clean", help="Truncate the selected log"):
                clear_logs(log_name)
                st.toast("Log cleared")

        log_text = read_log_tail(log_name, int(max_lines))
        render_log_box(log_text, height=560)

        if st.button("Clean ALL logs", type="secondary"):
            clear_logs(None)
            st.toast("All logs cleared")

        if refresh_s and refresh_s > 0:
            time.sleep(int(refresh_s))
            st.rerun()
