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

# RL replay buffer (self-play shards)
RL_BUFFER_DIR = "E:/chronos/chronos_rl_buffer"
os.makedirs(RL_BUFFER_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT,'data'), exist_ok=True)

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
