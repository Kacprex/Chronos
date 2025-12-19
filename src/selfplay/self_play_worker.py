import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
import time
from datetime import datetime
from typing import List, Tuple
import random
import torch
import chess
import numpy as np
from src.config import BEST_MODEL_PATH, RL_BUFFER_DIR
from src.nn.network import ChessNet
from src.nn.encoding import encode_board, move_to_index, MOVE_SPACE
from src.mcts.mcts import MCTS
from src.selfplay.opening_book import play_random_opening


def fmt_time(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def play_single_game(
    model,
    device,
    simulations: int,
    temperature_moves: int = 10,
    max_moves: int = 512,
):
    """
    Plays a single self-play game and returns training samples:
    (board_tensor, policy, value)

    - Uses temperature > 1.0 for first N moves
    - Applies Dirichlet noise at root
    - Correctly assigns value targets
    """

    import chess
    import random
    from src.mcts.mcts import MCTS
    from src.selfplay.opening_book import play_random_opening
    from src.nn.encoding import encode_board

    board = chess.Board()
    history = []  # (board_tensor, policy, was_white_turn)

    # Play a short random opening line from a small curated book to improve early-game diversity.
    # We do NOT record training samples for these forced opening plies (they are not MCTS-derived policies).
    play_random_opening(board, max_plies=6)

    move_count = board.ply()

    # Reuse a single MCTS instance per game (no tree reuse; run() builds a fresh root each call)
    mcts = MCTS(
        model=model,
        device=device,
        simulations=simulations,
        add_dirichlet_noise=True,
    )

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        move_count += 1

        moves, probs = mcts.run(board, move_count)
        if not moves:
            break


        move = np.random.choice(moves, p=probs)


        # store BEFORE move
        board_tensor = torch.from_numpy(encode_board(board))
        pi = np.zeros(MOVE_SPACE, dtype=np.float32)
        for m, p in zip(moves, probs):
            idx = move_to_index(m)
            if idx is not None:
                pi[idx] = p
        
        history.append((board_tensor, pi, board.turn))
        

        board.push(move)


    # Game result
    result = board.result(claim_draw=True)
    if result == "1-0":
        z_white = 1.0
    elif result == "0-1":
        z_white = -1.0
    else:
        z_white = 0.0

    samples = []
    for board_tensor, policy, was_white_turn in history:
        z = z_white if was_white_turn else -z_white
        samples.append((board_tensor, policy, z))

    return samples



def self_play(num_games: int = 50, simulations: int = 200, shard_size: int = 10_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Self-play using device: {device}")

    os.makedirs(RL_BUFFER_DIR, exist_ok=True)

    model = ChessNet().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    # find next shard id
    existing = [f for f in os.listdir(RL_BUFFER_DIR) if f.startswith("rl_shard_") and f.endswith(".pt")]
    if existing:
        existing.sort()
        last = existing[-1]
        shard_id = int(last.split("_")[-1].split(".")[0]) + 1
    else:
        shard_id = 1

    boards_list: List[torch.Tensor] = []
    policies_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []

    total_positions = 0
    start_time = time.time()
    for g in range(1, num_games + 1):
        game_start = time.time()
        samples = play_single_game(
            model=model,
            device=device,
            simulations=simulations
        )


        game_time = time.time() - game_start

        for b, pi, z in samples:
            boards_list.append(b)
            policies_list.append(torch.tensor(pi, dtype=torch.float32))
            values_list.append(torch.tensor([z], dtype=torch.float32))
            total_positions += 1

        log(f"Game {g}/{num_games} finished: positions={len(samples)}, "
            f"total_positions={total_positions}, time={fmt_time(game_time)}")

        if len(boards_list) >= shard_size:
            shard_path = os.path.join(RL_BUFFER_DIR, f"rl_shard_{shard_id:06d}.pt")
            log(f"Saving RL shard {shard_id} → {shard_path} "
                f"(positions={len(boards_list)}, total={total_positions})")
            torch.save({
                "boards": torch.stack(boards_list),
                "policies": torch.stack(policies_list),
                "values": torch.stack(values_list),
            }, shard_path)

            boards_list.clear()
            policies_list.clear()
            values_list.clear()
            shard_id += 1

        # periodic ETA logging
        elapsed = time.time() - start_time
        games_per_sec = g / elapsed if elapsed > 0 else 0
        games_left = num_games - g
        eta = games_left / games_per_sec if games_per_sec > 0 else 0

        log(f"Progress: {g}/{num_games} games "
            f"({g/num_games:.2%}), elapsed={fmt_time(elapsed)}, ETA≈{fmt_time(eta)}")

    # final shard
    if boards_list:
        shard_path = os.path.join(RL_BUFFER_DIR, f"rl_shard_{shard_id:06d}.pt")
        log(f"Saving final RL shard {shard_id} → {shard_path} "
            f"(positions={len(boards_list)}, total={total_positions})")
        torch.save({
            "boards": torch.stack(boards_list),
            "policies": torch.stack(policies_list),
            "values": torch.stack(values_list),
        }, shard_path)

    total_time = time.time() - start_time
    log(f"✔ Self-play complete: games={num_games}, positions={total_positions}, time={fmt_time(total_time)}")


if __name__ == "__main__":
    # Example: generate 100 games, 200 sims each
    self_play(num_games=100, simulations=200)
