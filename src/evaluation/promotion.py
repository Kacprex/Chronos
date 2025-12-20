import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
import json
import time
from datetime import datetime

import torch
import chess
import numpy as np

from src.config import BEST_MODEL_PATH, LATEST_MODEL_PATH
from src.nn.network import ChessNet
from src.mcts.mcts import MCTS
def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def play_game(model_white, model_black, device, simulations=200, temp_moves=0, max_moves=512) -> float:
    """Return 1.0 if white wins, 0.5 draw, 0.0 black wins."""
    board = chess.Board()

    mcts_white = MCTS(model_white, device=device, simulations=simulations)
    mcts_black = MCTS(model_black, device=device, simulations=simulations)

    move_count = 0
    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        if board.turn == chess.WHITE:
            mcts = mcts_white
        else:
            mcts = mcts_black

        moves, probs = mcts.run(board, move_number=move_count, add_noise=(move_count < temp_moves))
        if not moves:
            break

        # Early moves can be stochastic (if temp_moves > 0); later moves deterministic.
        if move_count < temp_moves:
            chosen_move = np.random.choice(moves, p=probs)
        else:
            chosen_move = moves[int(np.argmax(probs))]

        board.push(chosen_move)
        move_count += 1

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return 0.0
    else:
        return 0.5


def evaluate_and_promote(
    num_games: int = 50,
    threshold: float = 0.55,
    simulations: int = 200,
    *,
    loop_iteration: int = 1,
    max_iterations: int = 1,
    selfplay_sims: int | None = None,
    sf_rating_depth: int = 12,
    sf_rating_games: int = 20,
    sf_rating_sims: int | None = None,
):
    """
    Evaluate latest vs best via MCTS games.

    Returns a dict:
      {
        "winrate": float,          # latest score in [0,1]
        "promoted": bool,
        "num_games": int,
        "simulations": int,
        "threshold": float,
      }

    Also (optionally) sends Discord webhook embeds if configured in src.config:
      - CHRONOS_PROMOTION_WEBHOOK for every promotion run
      - CHRONOS_RATING_WEBHOOK only when promoted (Stockfish-based rating estimate)
    """
    # Import config as a module so we can gracefully handle older/newer
    # variable names without hard-failing on ImportError.
    import src.config as cfg

    DISCORD_PROMOTION_WEBHOOK = getattr(cfg, "DISCORD_PROMOTION_WEBHOOK", "")
    # Backward/forward compatibility: some versions used "RATING", others "RANKING".
    DISCORD_RATING_WEBHOOK = (
        getattr(cfg, "DISCORD_RATING_WEBHOOK", "")
        or getattr(cfg, "DISCORD_RANKING_WEBHOOK", "")
    )
    RATING_CACHE_PATH = getattr(cfg, "RATING_CACHE_PATH", "")
    ENGINE_PATH = getattr(cfg, "ENGINE_PATH", "")
    from src.logging.discord_webhooks import send_promotion_embed, send_rating_embed
    from src.evaluation.chronos_rating import estimate_chronos_rating_vs_stockfish

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Promotion eval using device: {device}")

    if not os.path.isfile(BEST_MODEL_PATH) or not os.path.isfile(LATEST_MODEL_PATH):
        log("Missing best or latest model file.")
        return {"winrate": 0.5, "promoted": False, "num_games": 0, "simulations": simulations, "threshold": threshold}

    best_model = ChessNet().to(device)
    latest_model = ChessNet().to(device)

    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    latest_model.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))

    best_model.eval()
    latest_model.eval()

    scores = []
    start_time = time.time()

    for g in range(1, num_games + 1):
        # Alternate colors for fairness
        if g % 2 == 1:
            score = play_game(latest_model, best_model, device, simulations=simulations)
        else:
            score = 1.0 - play_game(best_model, latest_model, device, simulations=simulations)
        scores.append(score)
        avg = sum(scores) / len(scores)
        log(f"Game {g}/{num_games}, score={score:.2f}, avg={avg:.3f}")

    final_score = sum(scores) / len(scores)
    log(f"Final score of latest vs best over {num_games} games: {final_score:.3f}")

    promoted = False
    if final_score >= threshold:
        promoted = True
        log(f"Latest model wins (>= {threshold:.2f}). Promoting to best.")
        torch.save(torch.load(LATEST_MODEL_PATH, map_location="cpu"), BEST_MODEL_PATH)
    else:
        log(f"Latest model not strong enough. Keeping current best.")

    log(f"Promotion eval finished in {time.time() - start_time:.1f}s")

    # --- Discord: promotion log (always) ---
    if DISCORD_PROMOTION_WEBHOOK:
        send_promotion_embed(
            DISCORD_PROMOTION_WEBHOOK,
            iteration=loop_iteration,
            max_iterations=max_iterations,
            promo_sims=simulations,
            winrate=float(final_score),
            promoted=bool(promoted),
            threshold=float(threshold),
            selfplay_sims=selfplay_sims,
        )

    # --- Discord: rating log (only if promoted) ---
    if promoted and DISCORD_RATING_WEBHOOK:
        # If not provided, reuse promotion sims (cheap & consistent)
        if sf_rating_sims is None:
            sf_rating_sims = simulations

        rr = estimate_chronos_rating_vs_stockfish(
            BEST_MODEL_PATH,
            depth=sf_rating_depth,
            num_games=sf_rating_games,
            simulations=sf_rating_sims,
        )
        log(f"SF rating estimate: score={rr.score:.3f}, elo_diff={rr.elo_diff:+.0f}, index={rr.rating_index:.0f} (depth={rr.depth}, games={rr.num_games}, {rr.seconds:.1f}s)")

        # load last rating index (if any) for delta coloring
        delta = None
        try:
            if os.path.isfile(RATING_CACHE_PATH):
                with open(RATING_CACHE_PATH, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                prev_idx = float(prev.get("rating_index"))
                delta = float(rr.rating_index - prev_idx)
        except Exception:
            delta = None

        # write current rating to cache
        try:
            os.makedirs(os.path.dirname(RATING_CACHE_PATH), exist_ok=True)
            with open(RATING_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rating_index": rr.rating_index,
                        "elo_diff": rr.elo_diff,
                        "score": rr.score,
                        "depth": rr.depth,
                        "num_games": rr.num_games,
                        "updated_utc": datetime.utcnow().isoformat() + "Z",
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            log(f"Warning: failed to write rating cache: {e}")

        send_rating_embed(
            DISCORD_RATING_WEBHOOK,
            rating_index=rr.rating_index,
            elo_diff=rr.elo_diff,
            score=rr.score,
            depth=rr.depth,
            num_games=rr.num_games,
            delta_vs_last=delta,
        )

    return {
        "winrate": float(final_score),
        "promoted": bool(promoted),
        "num_games": int(num_games),
        "simulations": int(simulations),
        "threshold": float(threshold),
    }




if __name__ == "__main__":
    evaluate_and_promote(num_games=50, threshold=0.55)
