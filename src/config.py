from pathlib import Path
import os

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Games
GM_GAMES_PATH = os.path.join(PROJECT_ROOT, "data","GM", "gm_games.csv")
AIVSAI = os.path.join(PROJECT_ROOT, "data","PGN", "ai_vs_ai_games.pgn")
SFVSAI = os.path.join(PROJECT_ROOT, "data","PGN", "sf_vs_ai_games.pgn")

# Where the processed shards will be stored (external disk!)
SHARD_DIR = "E:/chronos/SL_shards"

# Where trained models are saved
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

PHASE1_MODEL_PATH = os.path.join(MODEL_DIR,"checkpoints", "sl_final.pth")
# Phase 3 paths
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")      
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.pth")  

# --- Generation tracking & model rollback ---
# Generation starts at 0 and is incremented whenever latest -> best promotion succeeds.
GENERATION_PATH = os.path.join(MODEL_DIR, "generation.txt")

# Keep up to N archived best models for rollback:
#   models/model_0.pth, models/model_1.pth, ...
ARCHIVED_MODELS_KEEP = int(os.environ.get("CHRONOS_ARCHIVED_MODELS_KEEP", "5"))

# RL replay buffer (self-play shards)
RL_BUFFER_DIR = "E:/chronos/chronos_rl_buffer"

# RL shard numbering persistence (monotonic counter)
RL_SHARD_COUNTER_PATH = os.path.join(RL_BUFFER_DIR, "rl_shard_counter.txt")

# RL replay buffer size cap (bytes). Default is 250 GB (your stated budget).
RL_BUFFER_MAX_GB = int(os.environ.get("CHRONOS_RL_BUFFER_MAX_GB", "250"))
RL_BUFFER_MAX_BYTES = RL_BUFFER_MAX_GB * (1024 ** 3)

os.makedirs(RL_BUFFER_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT,'data'), exist_ok=True)

# RL training resume checkpoint
# Used by the hub and training code to resume RL training after interruptions.
RL_RESUME_PATH = os.path.join(MODEL_DIR, "checkpoints", "rl_resume.pt")

# If True, whenever a promotion attempt fails, the hub may optionally reset
# latest_model.pth back to best_model.pth (to avoid training drift). Kept False
# by default so training can continue from the current latest.
RESET_LATEST_ON_FAILED_PROMOTION = False

#Engine
ENGINE_PATH = os.path.join(PROJECT_ROOT, "engine", "stockfish.exe")

# Discord webhook logging (set as environment variables; do NOT hardcode secrets)
# Optional: Discord webhooks for logging ratings/promotions.
# Recommended: set as environment variables (don't hardcode secrets).
#
# Supported env var names:
# - CHRONOS_PROMOTION_WEBHOOK
# - CHRONOS_RANKING_WEBHOOK  (new name)
# - CHRONOS_RATING_WEBHOOK   (legacy name)
DISCORD_PROMOTION_WEBHOOK = os.environ.get('CHRONOS_PROMOTION_WEBHOOK', '').strip()

# "ranking" vs "rating" naming has drifted across versions; keep both for compatibility.
DISCORD_RANKING_WEBHOOK = os.environ.get('CHRONOS_RANKING_WEBHOOK', '').strip()
DISCORD_RATING_WEBHOOK = os.environ.get('CHRONOS_RATING_WEBHOOK', '').strip()

if not DISCORD_RANKING_WEBHOOK and DISCORD_RATING_WEBHOOK:
    DISCORD_RANKING_WEBHOOK = DISCORD_RATING_WEBHOOK
if not DISCORD_RATING_WEBHOOK and DISCORD_RANKING_WEBHOOK:
    DISCORD_RATING_WEBHOOK = DISCORD_RANKING_WEBHOOK

# Local rating cache file (kept in gitignored data/ by default)
RATING_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'chronos_rating_last.json')


# General logging (errors, status updates) - can be set via env or a local file.
# Priority: CHRONOS_LOG_WEBHOOK -> CHRONOS_DISCORD_WEBHOOK -> data/discord_webhook_url.txt
_LOG_HOOK = os.environ.get("CHRONOS_LOG_WEBHOOK", "").strip() or os.environ.get("CHRONOS_DISCORD_WEBHOOK", "").strip()
if not _LOG_HOOK:
    try:
        _LOG_HOOK = Path("data/discord_webhook_url.txt").read_text(encoding="utf-8").strip()
    except Exception:
        _LOG_HOOK = ""
DISCORD_LOG_WEBHOOK = _LOG_HOOK

