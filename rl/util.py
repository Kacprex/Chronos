from __future__ import annotations

import json
import os
from pathlib import Path
from time import time
from typing import Any, Dict

def default_root() -> Path:
    return Path(os.environ.get("CHRONOS_DATA_ROOT", r"E:/chronos"))

def now_ms() -> int:
    return int(time() * 1000)

def append_event(root: Path, obj: Dict[str, Any]) -> None:
    p = root / "logs" / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
