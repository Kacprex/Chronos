from __future__ import annotations

from datetime import datetime
from typing import Optional

def log(msg: str, prefix: Optional[str] = None) -> str:
    """Print a timestamped log line. Returns the formatted line."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}" if not prefix else f"[{now}] [{prefix}] {msg}"
    print(line)
    return line
