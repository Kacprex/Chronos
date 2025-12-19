import os
import re
import random
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
import chess

from src.config import RL_BUFFER_DIR, BEST_MODEL_PATH
from src.nn.network import ChessNet
from src.nn.encoding import encode_board, move_to_index, MOVE_SPACE
from src.mcts.mcts import MCTS


# ---------------------------
# RL buffer retention control
# ---------------------------
# Keep only the most recent N RL shards in RL_BUFFER_DIR to prevent disk from filling up.
# Override with environment variable CHRONOS_MAX_RL_SHARDS (e.g., 200, 300, 500).
MAX_RL_SHARDS = int(os.environ.get("CHRONOS_MAX_RL_SHARDS", "300"))
_RL_SHARD_RE = re.compile(r"^rl_shard_(\d{6})\.pt$")


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _next_shard_id() -> int:
    _ensure_dir(RL_BUFFER_DIR)
    best = 0
    for fn in os.listdir(RL_BUFFER_DIR):
        m = _RL_SHARD_RE.match(fn)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def prune_old_rl_shards(keep_last: int = MAX_RL_SHARDS) -> None:
    """
    Keep only the newest `keep_last` RL shards in RL_BUFFER_DIR.
    Only deletes files matching `rl_shard_XXXXXX.pt`.
    """
    if keep_last <= 0:
        return
    if not os.path.isdir(RL_BUFFER_DIR):
        return

    shards = []
    for fn in os.listdir(RL_BUFFER_DIR):
        m = _RL_SHARD_RE.match(fn)
        if not m:
            continue
        shard_id = int(m.group(1))
        shards.append((shard_id, os.path.join(RL_BUFFER_DIR, fn)))

    if len(shards) <= keep_last:
        return

    shards.sort(key=lambda x: x[0])  # oldest first by shard id
    for shard_id, path in shards[:-keep_last]:
        try:
            os.remove(path)
            log(f"Pruned old RL shard: {os.path.basename(path)}")
        except Exception as e:
            log(f"Warning: failed to delete {path}: {e}")


def _safe_move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    """
    Supports both signatures:
      move_to_index(move)
      move_to_index(move, board)
    """
    try:
        return move_to_index(move)  # type: ignore
    except TypeError:
        try:
            return move_to_index(move, board)  # type: ignore
        except Exception:
            return None
    except Exception:
        return None


def _result_to_value_white(result: str) -> float:
    r = str(result).strip()
    if r == "1-0":
        return 1.0
    if r == "0-1":
        return -1.0
    if r == "1/2-1/2":
        return 0.0
    return 0.0


def play_single_game(
    model: ChessNet,
    device: torch.device,
    simulations: int,
    temperature_plies: int = 10,
    initial_temperature: float = 1.25,
    max_moves: int = 512,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    """
    Plays one self-play game and returns:
      - boards: list of (18,8,8) float32 tensors (position BEFORE move)
      - policies: list of (MOVE_SPACE,) float32 tensors (MCTS distribution)
      - z_white: terminal outcome from WHITE perspective in [-1,1]
    """
    board = chess.Board()

    mcts = MCTS(
        model=model,
        device=device,
        simulations=simulations,
        cpuct=1.5,
        add_dirichlet_noise=True,
    )

    # IMPORTANT: your MCTS.run() does NOT accept a `temperature=` kwarg.
    # It uses internal fields: temp_initial / temp_moves. Set them here.
    if hasattr(mcts, "temp_initial"):
        mcts.temp_initial = float(initial_temperature)
    if hasattr(mcts, "temp_moves"):
        mcts.temp_moves = int(temperature_plies)

    boards_t: List[torch.Tensor] = []
    pis_t: List[torch.Tensor] = []

    for ply in range(1, max_moves + 1):
        if board.is_game_over(claim_draw=True):
            break

        # add Dirichlet noise only during exploration window
        add_noise = (ply <= temperature_plies)

        moves, probs = mcts.run(
            board,
            move_number=ply,
            add_noise=add_noise,
        )

        if not moves or probs is None or len(moves) != len(probs):
            break

        # Full MOVE_SPACE policy vector
        pi = torch.zeros(MOVE_SPACE, dtype=torch.float32)
        mass = 0.0
        for mv, p in zip(moves, probs):
            idx = _safe_move_to_index(board, mv)
            if idx is None:
                continue
            if p <= 0 or not np.isfinite(p):
                continue
            pi[idx] += float(p)
            mass += float(p)

        if mass <= 0.0 or not torch.isfinite(pi).all():
            # fallback: uniform over encodable moves
            enc = []
            for mv in moves:
                idx = _safe_move_to_index(board, mv)
                if idx is not None:
                    enc.append(idx)
            if not enc:
                break
            pi = torch.zeros(MOVE_SPACE, dtype=torch.float32)
            val = 1.0 / len(enc)
            for idx in enc:
                pi[idx] = val
        else:
            pi /= pi.sum().clamp_min(1e-12)

        # Save sample BEFORE move
        bt = torch.from_numpy(encode_board(board)).float()
        boards_t.append(bt)
        pis_t.append(pi)

        # Choose move:
        if ply > temperature_plies:
            # deterministic phase
            choice = int(np.argmax(probs))
        else:
            # exploration: sample
            probs_np = np.asarray(probs, dtype=np.float64)
            probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=0.0, neginf=0.0)
            s = probs_np.sum()
            probs_np = (np.ones_like(probs_np) / len(probs_np)) if s <= 0 else (probs_np / s)
            choice = int(np.random.choice(len(moves), p=probs_np))

        board.push(moves[choice])

    z_white = _result_to_value_white(board.result(claim_draw=True))
    return boards_t, pis_t, z_white


def self_play(
    num_games: int = 100,
    simulations: int = 200,
    shard_size: int = 10_000,
    temperature_plies: int = 10,
    initial_temperature: float = 1.25,
    seed: int = 42,
):
    """
    Generate self-play data and store it as RL shards in RL_BUFFER_DIR.

    Shard format:
      - boards: (N, 18, 8, 8)
      - policies: (N, MOVE_SPACE)
      - values: (N, 1)  (from side-to-move perspective)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Self-play device: {device.type}")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"BEST model not found: {BEST_MODEL_PATH}")

    model = ChessNet().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    log(f"Loaded BEST model: {BEST_MODEL_PATH}")

    _ensure_dir(RL_BUFFER_DIR)
    shard_id = _next_shard_id()

    boards_buf: List[torch.Tensor] = []
    pis_buf: List[torch.Tensor] = []
    vals_buf: List[torch.Tensor] = []

    def flush():
        nonlocal shard_id
        if not boards_buf:
            return

        path = os.path.join(RL_BUFFER_DIR, f"rl_shard_{shard_id:06d}.pt")
        torch.save(
            {
                "boards": torch.stack(boards_buf),
                "policies": torch.stack(pis_buf),
                "values": torch.stack(vals_buf),
            },
            path,
        )
        log(f"Saved RL shard {shard_id} -> {path} (N={len(boards_buf)})")
        shard_id += 1

        boards_buf.clear()
        pis_buf.clear()
        vals_buf.clear()

        # Prevent disk from filling up overnight
        prune_old_rl_shards(keep_last=MAX_RL_SHARDS)

    for g in range(1, num_games + 1):
        b_list, pi_list, z_white = play_single_game(
            model=model,
            device=device,
            simulations=simulations,
            temperature_plies=temperature_plies,
            initial_temperature=initial_temperature,
        )

        for bt, pi in zip(b_list, pi_list):
            # plane 12 is side-to-move; 1 means white-to-move
            stm_is_white = bool(bt[12].max().item() > 0.5)
            z = z_white if stm_is_white else -z_white

            boards_buf.append(bt)
            pis_buf.append(pi)
            vals_buf.append(torch.tensor([z], dtype=torch.float32))

            if len(boards_buf) >= shard_size:
                flush()

        if g % 10 == 0:
            log(f"Self-play progress: {g}/{num_games} games")

    flush()
    log("Self-play complete.")


if __name__ == "__main__":
    self_play()
