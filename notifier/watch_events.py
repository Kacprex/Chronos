from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

from notifier.discord import RateLimiter, send_webhook


def default_root() -> Path:
    return Path(os.environ.get("CHRONOS_DATA_ROOT", r"E:/chronos"))


def state_path(root: Path) -> Path:
    return root / "notifier" / "state.json"


def load_state(root: Path) -> Dict[str, Any]:
    p = state_path(root)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(root: Path, st: Dict[str, Any]) -> None:
    p = state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def should_notify(ev: Dict[str, Any]) -> bool:
    t = ev.get("type", "")
    return t in {"pretrain_start", "pretrain_epoch_end", "pretrain_done", "engine_start"}


def format_msg(ev: Dict[str, Any]) -> str:
    t = ev.get("type", "")
    run = ev.get("run_id", "?")
    if t == "pretrain_epoch_end":
        loss = ev.get("loss", 0.0)
        try:
            loss = float(loss)
        except Exception:
            loss = 0.0
        return f"🕰️ Chronos pretrain `{run}` epoch {ev.get('epoch')} loss={loss:.4f}"
    if t == "pretrain_done":
        return f"✅ Chronos pretrain `{run}` finished. ckpt={ev.get('ckpt')}"
    if t == "pretrain_start":
        return f"▶️ Chronos pretrain `{run}` started. shards={ev.get('shards')}"
    if t == "engine_start":
        return "♟️ Chronos engine started."
    return f"Chronos event {t} ({run})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhook", default=os.environ.get("CHRONOS_DISCORD_WEBHOOK", ""), help="Discord webhook URL")
    ap.add_argument("--root", default=str(default_root()), help="Data root")
    ap.add_argument("--poll", type=float, default=1.0, help="Polling interval seconds")
    ap.add_argument("--rate", type=float, default=1.0, help="Min seconds between webhook posts")
    args = ap.parse_args()

    webhook = args.webhook.strip()
    if not webhook:
        raise SystemExit("No webhook URL provided (use --webhook or CHRONOS_DISCORD_WEBHOOK).")

    root = Path(args.root)
    ev_path = root / "logs" / "events.jsonl"
    ev_path.parent.mkdir(parents=True, exist_ok=True)

    stt = load_state(root)
    offset = int(stt.get("offset", 0))

    rl = RateLimiter(min_interval_s=float(args.rate))

    print(f"Watching: {ev_path}")
    print(f"Starting offset: {offset}")

    while True:
        if not ev_path.exists():
            time.sleep(args.poll)
            continue

        size = ev_path.stat().st_size
        if offset > size:
            offset = 0

        with ev_path.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
            offset = f.tell()

        if chunk:
            for ln in chunk.splitlines():
                try:
                    ev = json.loads(ln.decode("utf-8"))
                except Exception:
                    continue

                if not should_notify(ev):
                    continue

                msg = format_msg(ev)
                rl.wait()
                ok, info = send_webhook(webhook, msg, username="Chronos")
                if not ok:
                    print("Webhook failed:", info)

            stt["offset"] = offset
            save_state(root, stt)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
