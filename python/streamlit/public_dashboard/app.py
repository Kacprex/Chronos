from __future__ import annotations

import html as _html
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = REPO_ROOT / "data" / "logs"


def list_logs() -> list[str]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.name for p in LOGS_DIR.glob("*.log")])


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


def render_log_box(text: str, height: int = 560) -> None:
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


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Chronos – Dashboard", layout="wide")

st.title("Chronos – Public Dashboard")
st.caption("Read-only log viewer (no controls).")

logs = list_logs()
if not logs:
    st.info("No logs found yet.")
    st.stop()

top = st.columns([3, 1, 1])
with top[0]:
    log_name = st.selectbox("Select log", logs, key="log_select_public")
with top[1]:
    max_lines = st.number_input("Tail lines", min_value=200, max_value=20000, value=3000, step=200)
with top[2]:
    refresh_s = st.selectbox("Auto-refresh", [0, 1, 2, 5, 10], index=2)

log_text = read_log_tail(log_name, int(max_lines))
render_log_box(log_text, height=600)

if refresh_s and refresh_s > 0:
    time.sleep(int(refresh_s))
    st.rerun()
