from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional

import torch


def _hash_tensor_row(t: torch.Tensor) -> str:
    """Stable hash for one encoded board tensor row."""
    # Use BLAKE2 for speed; include dtype/shape implicitly via bytes.
    h = hashlib.blake2b(t.contiguous().numpy().tobytes(), digest_size=16)
    return h.hexdigest()


def analyze_rl_shard(
    shard: Dict[str, Any],
    sample_limit: int = 5000,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute lightweight quality stats for a RL shard.

    Expected shard keys: x (boards), pi (policy probs), z (outcome values).
    Older/alternate keys are handled in train_rl.py before calling this.
    """
    x = shard["x"]
    pi = shard.get("pi")
    z = shard.get("z")

    n = int(x.shape[0])
    k = min(n, int(sample_limit))

    # Sample evenly across the shard to avoid only looking at early positions.
    if k <= 0:
        return {
            "n": n,
            "sample": 0,
            "unique": 0,
            "unique_ratio": 0.0,
        }

    idx = torch.linspace(0, n - 1, steps=k).long()
    # Hash boards
    hashes = set()
    for i in idx.tolist():
        hashes.add(_hash_tensor_row(x[i]))
    unique = len(hashes)
    unique_ratio = unique / k

    out: Dict[str, Any] = {
        "n": n,
        "sample": k,
        "unique": unique,
        "unique_ratio": float(unique_ratio),
    }

    # Policy entropy (higher = more diverse / less deterministic). This is not
    # "good" by itself, but helps diagnose collapse.
    if pi is not None and isinstance(pi, torch.Tensor) and pi.numel() > 0:
        pi_s = pi[idx]
        # clamp to avoid log(0)
        pi_s = torch.clamp(pi_s, min=eps)
        ent = -(pi_s * torch.log(pi_s)).sum(dim=1)
        out["pi_entropy_mean"] = float(ent.mean().item())
        out["pi_entropy_std"] = float(ent.std(unbiased=False).item())

    # Outcome distribution
    if z is not None and isinstance(z, torch.Tensor) and z.numel() > 0:
        z_s = z[idx].float()
        out["z_mean"] = float(z_s.mean().item())
        out["z_std"] = float(z_s.std(unbiased=False).item())
        out["z_draw_frac"] = float((z_s.abs() < 1e-6).float().mean().item())
        out["z_win_frac"] = float((z_s > 0.5).float().mean().item())
        out["z_loss_frac"] = float((z_s < -0.5).float().mean().item())

    return out
