import os
import time
import random
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
import chess

from src.config import BEST_MODEL_PATH, PHASE1_MODEL_PATH, RL_BUFFER_DIR
from src.nn.network import ChessNet
from src.nn.encoding import encode_board, move_to_index, MOVE_SPACE
from src.mcts.mcts import MCTS


def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _safe_move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    """Supports both signatures move_to_index(move) and move_to_index(move, board)."""
    try:
        return move_to_index(move)  # type: ignore
    except TypeError:
        try:
            return move_to_index(move, board)  # type: ignore
        except Exception:
            return None
    except Exception:
        return None


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """AlphaZero-style temperature on a probability vector."""
    temperature = float(temperature)
    if temperature <= 0.0:
        out = np.zeros_like(probs)
        out[int(np.argmax(probs))] = 1.0
        return out

    # Sharpen / flatten
    p = np.clip(probs, 1e-12, 1.0) ** (1.0 / temperature)
    p = p / np.sum(p)
    return p


def _load_model(device: torch.device) -> ChessNet:
    model = ChessNet().to(device)

    # Prefer BEST_MODEL_PATH, fall back to PHASE1_MODEL_PATH (SL final)
    if os.path.exists(BEST_MODEL_PATH):
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt)
        _log(f"Loaded BEST model: {BEST_MODEL_PATH}")
    elif os.path.exists(PHASE1_MODEL_PATH):
        ckpt = torch.load(PHASE1_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt)
        _log(f"Loaded SL model (PHASE1): {PHASE1_MODEL_PATH}")
    else:
        _log("WARNING: No model checkpoint found. Using randomly initialized weights.")

    model.eval()
    return model


def play_single_game(
    model: ChessNet,
    device: torch.device,
    simulations: int = 200,
    temperature_moves: int = 10,
    max_moves: int = 512,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Run one self-play game (MCTS guided) and return training samples:

        (board_tensor, policy_target, value_target)

    - board_tensor: (18, 8, 8) float32
    - policy_target: (MOVE_SPACE,) float32 probabilities from MCTS visits
    - value_target: float in [-1, 1], from the perspective of the side-to-move at that position

    Notes:
    - Temperature > 1.0 early to increase exploration, then low temperature later.
    - MCTS adds Dirichlet noise at root when add_noise=True.
    """
    board = chess.Board()

    # store per-position samples before pushing the selected move:
    # (board_tensor, policy_vec, was_white_turn)
    history: List[Tuple[torch.Tensor, torch.Tensor, bool]] = []

    for ply in range(1, max_moves + 1):
        if board.is_game_over(claim_draw=True):
            break

        # Temperature schedule
        temperature = 1.25 if ply <= temperature_moves else 0.10

        # Fresh MCTS each move (simple + safe)
        mcts = MCTS(
            model=model,
            device=device,
            simulations=simulations,
            cpuct=1.5,
            add_dirichlet_noise=True,
        )

        moves, probs = mcts.run(board, move_number=ply, add_noise=True)
        if not moves or probs is None or len(moves) != len(probs):
            break

        probs = _apply_temperature(probs.astype(np.float64), temperature)

        # Build dense policy target over MOVE_SPACE indices
        policy_vec = torch.zeros(MOVE_SPACE, dtype=torch.float32)
        mass = 0.0
        for mv, p in zip(moves, probs):
            idx = _safe_move_to_index(board, mv)
            if idx is None:
                continue
            policy_vec[idx] += float(p)
            mass += float(p)

        # If some moves were un-encodable, renormalize what remains
        if mass > 1e-8:
            policy_vec /= float(policy_vec.sum())

        board_tensor = torch.from_numpy(encode_board(board)).float()
        history.append((board_tensor, policy_vec, board.turn == chess.WHITE))

        # Sample the move according to probs (after temperature)
        choice = int(np.random.choice(len(moves), p=probs))
        board.push(moves[choice])

    # Game outcome from WHITE perspective
    res = board.result(claim_draw=True)
    if res == "1-0":
        z_white = 1.0
    elif res == "0-1":
        z_white = -1.0
    else:
        z_white = 0.0

    # Convert to side-to-move perspective per stored position
    out: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    for bt, pi, was_white_turn in history:
        z = z_white if was_white_turn else -z_white
        out.append((bt, pi, float(z)))

    return out


def _next_shard_id() -> int:
    os.makedirs(RL_BUFFER_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RL_BUFFER_DIR) if f.startswith("rl_shard_") and f.endswith(".pt")]
    if not existing:
        return 1
    existing.sort()
    last = existing[-1]
    try:
        return int(last.split("_")[-1].split(".")[0]) + 1
    except Exception:
        return 1


def _save_rl_shard(shard_id: int, boards: List[torch.Tensor], policies: List[torch.Tensor], values: List[torch.Tensor]) -> None:
    path = os.path.join(RL_BUFFER_DIR, f"rl_shard_{shard_id:06d}.pt")
    os.makedirs(RL_BUFFER_DIR, exist_ok=True)

    torch.save(
        {
            "boards": torch.stack(boards),      # (N, 18, 8, 8)
            "policies": torch.stack(policies),  # (N, MOVE_SPACE)
            "values": torch.stack(values),      # (N, 1)
        },
        path,
    )
    _log(f"Saved RL shard {shard_id} → {path} (N={len(boards)})")


def self_play(
    num_games: int = 50,
    simulations: int = 200,
    shard_size: int = 10_000,
    temperature_moves: int = 10,
    max_moves: int = 512,
) -> None:
    """
    Generate self-play data and write RL shards to RL_BUFFER_DIR.

    Each shard contains:
      - boards:   (N, 18, 8, 8)
      - policies: (N, MOVE_SPACE)  (MCTS visit distribution)
      - values:   (N, 1)           (terminal outcome, side-to-move perspective)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Self-play device: {device}")
    model = _load_model(device)

    shard_id = _next_shard_id()

    boards: List[torch.Tensor] = []
    policies: List[torch.Tensor] = []
    values: List[torch.Tensor] = []

    total_positions = 0
    t0 = time.time()

    for g in range(1, num_games + 1):
        samples = play_single_game(
            model=model,
            device=device,
            simulations=simulations,
            temperature_moves=temperature_moves,
            max_moves=max_moves,
        )

        for bt, pi, z in samples:
            boards.append(bt)
            policies.append(pi)
            values.append(torch.tensor([z], dtype=torch.float32))
            total_positions += 1

            if len(boards) >= shard_size:
                _save_rl_shard(shard_id, boards, policies, values)
                shard_id += 1
                boards.clear()
                policies.clear()
                values.clear()

        if g % max(1, num_games // 10) == 0 or g == num_games:
            elapsed = time.time() - t0
            pps = total_positions / elapsed if elapsed > 0 else 0.0
            _log(f"Games {g}/{num_games} | positions={total_positions} | {pps:.1f} pos/s | elapsed={_fmt_time(elapsed)}")

    if boards:
        _save_rl_shard(shard_id, boards, policies, values)

    elapsed = time.time() - t0
    _log(f"Self-play complete. Games={num_games}, positions={total_positions}, time={_fmt_time(elapsed)}")


if __name__ == "__main__":
    # sensible defaults for quick testing
    self_play(num_games=10, simulations=200, shard_size=10_000)
