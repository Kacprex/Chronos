import os
import time
from datetime import datetime
from typing import List, Tuple

import chess
import numpy as np
import torch

from src.config import BEST_MODEL_PATH, RL_BUFFER_DIR
from src.mcts.mcts import MCTS
from src.nn.encoding import MOVE_SPACE, encode_board, move_to_index
from src.nn.network import ChessNet


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def _safe_probs(probs: np.ndarray) -> np.ndarray:
    """Normalize probabilities and guard against NaNs."""
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return p.astype(np.float32)
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        p = np.ones_like(p, dtype=np.float64) / len(p)
    else:
        p = p / s
    return p.astype(np.float32)


def play_single_game(
    model: torch.nn.Module,
    device: torch.device,
    simulations: int,
    temperature_moves: int = 20,
    max_moves: int = 512,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate one self-play game and return training samples.

    Each sample is (board_planes, pi, z), where:
      - board_planes: (18,8,8) float32 tensor
      - pi: (MOVE_SPACE,) float32 tensor (MCTS visit distribution)
      - z: (1,) float32 tensor in [-1,1] (outcome from side-to-move perspective)
    """

    board = chess.Board()
    history: List[Tuple[torch.Tensor, torch.Tensor, bool]] = []  # (board_tensor, pi_tensor, was_white_turn)

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        ply += 1

        # Fresh MCTS per move (simple + safe)
        mcts = MCTS(
            model=model,
            device=device,
            simulations=simulations,
            cpuct=1.5,
            add_dirichlet_noise=True,
        )
        # Align temperature schedule with self-play settings
        mcts.temp_initial = 1.25
        mcts.temp_moves = int(temperature_moves)

        moves, probs = mcts.run(board, move_number=ply, add_noise=True)
        if not moves or probs is None:
            break

        probs = _safe_probs(probs)

        # Sample a move according to MCTS distribution
        choice = int(np.random.choice(len(moves), p=probs))
        move = moves[choice]

        # Build training target policy vector
        pi = np.zeros(MOVE_SPACE, dtype=np.float32)
        for m, p in zip(moves, probs):
            idx = move_to_index(m)
            if idx is not None:
                pi[idx] += float(p)

        # If encoding dropped too many moves, fall back to chosen move.
        s = float(pi.sum())
        if not np.isfinite(s) or s <= 0:
            idx = move_to_index(move)
            if idx is not None:
                pi[idx] = 1.0
            else:
                # total fallback: uniform
                pi[:] = 1.0 / MOVE_SPACE
        else:
            pi /= s

        board_tensor = torch.from_numpy(encode_board(board)).to(torch.float32)
        pi_tensor = torch.from_numpy(pi).to(torch.float32)
        history.append((board_tensor, pi_tensor, board.turn == chess.WHITE))

        board.push(move)

    # Terminal result from White's perspective
    result = board.result(claim_draw=True)
    if result == "1-0":
        z_white = 1.0
    elif result == "0-1":
        z_white = -1.0
    else:
        z_white = 0.0

    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for board_tensor, pi_tensor, was_white_turn in history:
        z = z_white if was_white_turn else -z_white
        samples.append((board_tensor, pi_tensor, torch.tensor([z], dtype=torch.float32)))

    return samples


def self_play(num_games: int = 50, simulations: int = 200, shard_size: int = 10_000) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Self-play device: {device}")

    os.makedirs(RL_BUFFER_DIR, exist_ok=True)

    model = ChessNet().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    _log(f"Loaded BEST model: {BEST_MODEL_PATH}")

    # Determine next shard id
    existing = [
        f for f in os.listdir(RL_BUFFER_DIR)
        if f.startswith("rl_shard_") and f.endswith(".pt")
    ]
    if existing:
        existing.sort()
        last_id = int(existing[-1].split("_")[-1].split(".")[0])
        shard_id = last_id + 1
    else:
        shard_id = 1

    boards_list: List[torch.Tensor] = []
    policies_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []

    total_positions = 0
    t0 = time.time()

    for g in range(1, num_games + 1):
        g0 = time.time()
        samples = play_single_game(
            model=model,
            device=device,
            simulations=simulations,
        )

        for b, pi, z in samples:
            boards_list.append(b)
            policies_list.append(pi)
            values_list.append(z)
            total_positions += 1

        _log(
            f"Game {g}/{num_games} finished: positions={len(samples)}, "
            f"total_positions={total_positions}, time={_fmt_time(time.time() - g0)}"
        )

        # Save shard when full
        if len(boards_list) >= shard_size:
            shard_path = os.path.join(RL_BUFFER_DIR, f"rl_shard_{shard_id:06d}.pt")
            _log(
                f"Saving RL shard {shard_id} → {shard_path} "
                f"(positions={len(boards_list)}, total={total_positions})"
            )
            torch.save(
                {
                    "boards": torch.stack(boards_list).to(torch.float32),
                    "policies": torch.stack(policies_list).to(torch.float32),
                    "values": torch.stack(values_list).to(torch.float32),
                },
                shard_path,
            )
            boards_list.clear()
            policies_list.clear()
            values_list.clear()
            shard_id += 1

        # ETA
        elapsed = time.time() - t0
        gps = g / elapsed if elapsed > 0 else 0.0
        eta = (num_games - g) / gps if gps > 0 else 0.0
        _log(f"Progress: {g}/{num_games} ({g/num_games:.1%}), elapsed={_fmt_time(elapsed)}, ETA≈{_fmt_time(eta)}")

    # Final shard
    if boards_list:
        shard_path = os.path.join(RL_BUFFER_DIR, f"rl_shard_{shard_id:06d}.pt")
        _log(
            f"Saving final RL shard {shard_id} → {shard_path} "
            f"(positions={len(boards_list)}, total={total_positions})"
        )
        torch.save(
            {
                "boards": torch.stack(boards_list).to(torch.float32),
                "policies": torch.stack(policies_list).to(torch.float32),
                "values": torch.stack(values_list).to(torch.float32),
            },
            shard_path,
        )

    _log(f"✔ Self-play complete: games={num_games}, positions={total_positions}, time={_fmt_time(time.time() - t0)}")


if __name__ == "__main__":
    self_play(num_games=100, simulations=200, shard_size=10_000)
