from __future__ import annotations
from pathlib import Path
import shutil
import json
from .paths import ensure_layout

def export_public_status(status: dict) -> None:
    p = ensure_layout()
    (p["public"] / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

def export_public_metrics(src_csv: Path, dest_name: str) -> None:
    p = ensure_layout()
    dest = p["public_metrics"] / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_csv, dest)

def export_latest_model(path_to_model: Path) -> None:
    p = ensure_layout()
    (p["public"] / "latest_model.txt").write_text(str(path_to_model), encoding="utf-8")
