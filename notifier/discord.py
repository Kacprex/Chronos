from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


@dataclass
class RateLimiter:
    min_interval_s: float = 1.0
    _last: float = 0.0

    def wait(self) -> None:
        now = time.time()
        dt = now - self._last
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last = time.time()


def send_webhook(url: str, content: str, username: Optional[str] = None) -> Tuple[bool, str]:
    payload = {"content": content}
    if username:
        payload["username"] = username
    try:
        r = requests.post(url, json=payload, timeout=10)
        if 200 <= r.status_code < 300:
            return True, "ok"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)
