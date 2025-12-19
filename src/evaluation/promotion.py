import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
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


def evaluate_and_promote(num_games: int = 50, threshold: float = 0.55):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Promotion eval using device: {device}")

    if not os.path.isfile(BEST_MODEL_PATH) or not os.path.isfile(LATEST_MODEL_PATH):
        log("Missing best or latest model file.")
        return

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
            score = play_game(latest_model, best_model, device)
        else:
            score = 1.0 - play_game(best_model, latest_model, device)

        scores.append(score)
        avg_score = sum(scores) / len(scores)
        log(f"Game {g}/{num_games}, score={score:.2f}, avg={avg_score:.3f}")

    final_score = sum(scores) / len(scores)
    log(f"Final score of latest vs best over {num_games} games: {final_score:.3f}")

    if final_score >= threshold:
        log(f"Latest model wins (>= {threshold:.2f}). Promoting to best.")
        torch.save(torch.load(LATEST_MODEL_PATH, map_location="cpu"), BEST_MODEL_PATH)
    else:
        log(f"Latest model not strong enough. Keeping current best.")

    log(f"Promotion eval finished in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    evaluate_and_promote(num_games=50, threshold=0.55)
