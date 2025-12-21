import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 10) -> None:
    if not url:
        return
    url = url.strip()
    # Normalize legacy Discord webhook domain to avoid redirects that can break POST.
    url = url.replace("https://discordapp.com/api/webhooks/", "https://discord.com/api/webhooks/")
    url = url.replace("http://discordapp.com/api/webhooks/", "https://discord.com/api/webhooks/")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Content-Type": "application/json",
            # Some endpoints reject requests without a UA.
            "User-Agent": "Chronos/1.0 (urllib)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        # Read body for better diagnostics (Discord often includes a JSON error).
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        print(f"[discord] Warning: webhook HTTP {e.code} {e.reason}. {body}".strip())
    except Exception as e:
        # Never crash training because Discord failed.
        print(f"[discord] Warning: failed to post webhook: {e}")


def send_discord_message(
    webhook_url: str,
    content: str,
    *,
    username: str = "Chronos",
    timeout: int = 10,
) -> None:
    """Send a plain text Discord webhook message.

    Discord has a ~2000 character content limit. We chunk long messages safely.
    """
    if not webhook_url or not content:
        return

    max_len = 1900  # keep some headroom
    chunks = [content[i : i + max_len] for i in range(0, len(content), max_len)]
    for chunk in chunks:
        _post_json(
            webhook_url,
            {
                "username": username,
                "content": chunk,
                # Avoid pinging anyone by accident
                "allowed_mentions": {"parse": []},
            },
            timeout=timeout,
        )


def send_promotion_embed(
    webhook_url: str,
    *,
    iteration: int,
    max_iterations: int,
    promo_sims: int,
    winrate: float,
    promoted: bool,
    threshold: float,
    selfplay_sims: Optional[int] = None,
) -> None:
    """
    Sends an embed about a promotion run.

    Color rules:
      - green if promoted
      - yellow if not promoted but winrate >= 0.48
      - red otherwise (regress)
    """
    if promoted:
        color = 0x2ECC71  # green
        status = "PROMOTED ✅"
    elif winrate >= 0.48:
        color = 0xF1C40F  # yellow
        status = "NO PROMOTION (close)"
    else:
        color = 0xE74C3C  # red
        status = "REGRESS ❌"

    fields = [
        {"name": "Loop", "value": f"{iteration}/{max_iterations}", "inline": True},
        {"name": "Promotion MCTS sims", "value": str(promo_sims), "inline": True},
        {"name": "Winrate (latest vs best)", "value": f"{winrate*100:.2f}%", "inline": True},
        {"name": "Threshold", "value": f"{threshold*100:.1f}%", "inline": True},
    ]
    if selfplay_sims is not None:
        fields.insert(1, {"name": "Self-play MCTS sims", "value": str(selfplay_sims), "inline": True})

    payload = {
        "embeds": [
            {
                "title": "Chronos — Promotion Result",
                "description": status,
                "color": color,
                "fields": fields,
                "timestamp": _utc_iso(),
            }
        ]
    }
    _post_json(webhook_url, payload)


def send_rating_embed(
    webhook_url: str,
    *,
    rating_index: float,
    elo_diff: float,
    score: float,
    depth: int,
    num_games: int,
    delta_vs_last: Optional[float] = None,
) -> None:
    """
    Sends an embed with a Stockfish-based rating estimate.

    Color rules:
      - green if improved vs last (delta > 0)
      - yellow if ~same (-5..+5)
      - red if worse (delta < 0)
      - if no last value: yellow
    """
    if delta_vs_last is None:
        color = 0xF1C40F
        trend = "Baseline"
    else:
        if delta_vs_last > 5:
            color = 0x2ECC71
            trend = f"Up +{delta_vs_last:.0f}"
        elif delta_vs_last < -5:
            color = 0xE74C3C
            trend = f"Down {delta_vs_last:.0f}"
        else:
            color = 0xF1C40F
            trend = f"Stable {delta_vs_last:+.0f}"

    payload = {
        "embeds": [
            {
                "title": "Chronos — SF Rating Update",
                "description": f"Trend: **{trend}**",
                "color": color,
                "fields": [
                    {"name": "Chronos rating index", "value": f"{rating_index:.0f}", "inline": True},
                    {"name": "Elo diff vs SF", "value": f"{elo_diff:+.0f}", "inline": True},
                    {"name": "Score vs SF", "value": f"{score:.3f} over {num_games} games", "inline": True},
                    {"name": "Stockfish depth", "value": str(depth), "inline": True},
                ],
                "timestamp": _utc_iso(),
            }
        ]
    }
    _post_json(webhook_url, payload)
