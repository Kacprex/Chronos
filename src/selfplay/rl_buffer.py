from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CleanupResult:
    removed_files: List[str]
    kept_files: List[str]
    total_bytes_before: int
    total_bytes_after: int


def _list_shards(buffer_dir: str, pattern: str = "rl_shard_*.pt") -> List[Path]:
    p = Path(buffer_dir)
    if not p.exists():
        return []

    shards = list(p.glob(pattern))
    # Oldest first (mtime, then name as tie-breaker)
    shards.sort(key=lambda fp: (fp.stat().st_mtime, fp.name))
    return shards


def _total_bytes(files: List[Path]) -> int:
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except OSError:
            # If the file disappears mid-iteration, ignore.
            pass
    return total


def cleanup_rl_buffer(
    buffer_dir: str,
    max_gb: float,
    *,
    min_keep_shards: int = 8,
    pattern: str = "rl_shard_*.pt",
) -> CleanupResult:
    """Keep the RL buffer bounded by total on-disk size.

    This deletes the oldest shards first until total size <= max_gb,
    while keeping at least min_keep_shards shards.

    If max_gb <= 0, no files are removed.
    """

    shards = _list_shards(buffer_dir, pattern=pattern)
    total_before = _total_bytes(shards)

    if max_gb is None or max_gb <= 0:
        return CleanupResult(
            removed_files=[],
            kept_files=[str(p) for p in shards],
            total_bytes_before=total_before,
            total_bytes_after=total_before,
        )

    limit_bytes = int(max_gb * 1024 * 1024 * 1024)

    removed: List[str] = []
    total = total_before

    # Delete oldest first while respecting min_keep_shards.
    while total > limit_bytes and len(shards) > max(0, min_keep_shards):
        oldest = shards.pop(0)
        try:
            size = oldest.stat().st_size
        except OSError:
            size = 0

        try:
            oldest.unlink(missing_ok=True)
            removed.append(str(oldest))
            total = max(0, total - size)
        except OSError:
            # If we cannot delete (locked), stop to avoid tight loops.
            shards.insert(0, oldest)
            break

    total_after = _total_bytes(shards)
    return CleanupResult(
        removed_files=removed,
        kept_files=[str(p) for p in shards],
        total_bytes_before=total_before,
        total_bytes_after=total_after,
    )
