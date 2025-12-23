from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from src.config import DISCORD_LOG_WEBHOOK
from src.logging.discord_webhooks import send_discord_message


@dataclass
class DiscordLogConfig:
    username: str = "Chronos"
    min_interval_s: float = 2.0
    max_len: int = 1900


_LAST_SENT_AT: float = 0.0
_LAST_SENT_FINGERPRINT: str = ""


_IMPORTANT_PATTERNS = [
    # Milestones (avoid sending every loop separator / status line)
    re.compile(r"Promotion eval"),
    re.compile(r"Promoted latest model"),
    re.compile(r"Latest model not strong enough"),
    re.compile(r"Saved latest RL model"),
    re.compile(r"RL training complete"),
    re.compile(r"Self-play complete"),
    re.compile(r"ERROR", re.IGNORECASE),
]


def _should_send(msg: str) -> bool:
    msg = msg.strip()
    if not msg:
        return False
    return any(p.search(msg) for p in _IMPORTANT_PATTERNS)


def log_to_discord(msg: str, *, cfg: Optional[DiscordLogConfig] = None) -> bool:
    """Send a log line to Discord (best-effort).

    - Only sends "important" events (to avoid rate limiting / spam)
    - Applies a simple cooldown + dedupe
    """
    global _LAST_SENT_AT, _LAST_SENT_FINGERPRINT

    if not DISCORD_LOG_WEBHOOK:
        return False

    if not _should_send(msg):
        return False

    cfg = cfg or DiscordLogConfig()
    now = time.time()

    # Cooldown
    if (now - _LAST_SENT_AT) < cfg.min_interval_s:
        return False

    # Dedupe
    fingerprint = msg.strip()
    if fingerprint == _LAST_SENT_FINGERPRINT:
        return False

    content = fingerprint
    if len(content) > cfg.max_len:
        content = content[: cfg.max_len - 3] + "..."

    ok = send_discord_message(DISCORD_LOG_WEBHOOK, content, username=cfg.username)
    if ok:
        _LAST_SENT_AT = now
        _LAST_SENT_FINGERPRINT = fingerprint
    return ok
