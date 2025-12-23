from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import DEBUG_ROOT


@dataclass(frozen=True)
class RunContext:
    """Holds paths for debug artifacts for a single run (RL loop / promote / train)."""

    run_dir: str

    @property
    def root(self) -> str:
        return self.run_dir

    def path(self, *parts: str) -> str:
        return str(Path(self.run_dir, *parts))

    def subdir(self, name: str) -> str:
        p = Path(self.run_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        return str(p)


def create_run_context(kind: str, enabled: bool = True) -> Optional[RunContext]:
    """Create a timestamped run directory under ./debug/<kind>/.

    Returns None if enabled=False.
    """
    if not enabled:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(DEBUG_ROOT) / kind
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_dir=str(run_dir))


def write_json(path: str, obj: Any) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, obj: Any) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
