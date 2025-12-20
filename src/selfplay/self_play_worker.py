import os
import multiprocessing as mp
from math import ceil
import time
from datetime import datetime
from typing import List, Tuple

import chess
import numpy as np
import torch

from src.config import BEST_MODEL_PATH, RL_BUFFER_DIR
from src.mcts.mcts import MCTS
from src.inference.batched_inference import BatchedInferenceServer, BatchedInferenceClient
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


def _discover_next_shard_id(rl_dir: str) -> int:
    """Return next shard id based on existing rl_shard_XXXXXX.pt files."""
    try:
        max_id = -1
        for name in os.listdir(rl_dir):
            if not name.startswith("rl_shard_") or not name.endswith(".pt"):
                continue
            stem = name[len("rl_shard_") : -len(".pt")]
            if stem.isdigit():
                max_id = max(max_id, int(stem))
        return max_id + 1
    except FileNotFoundError:
        return 0


def _save_shard(samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], shard_id: int) -> str:
    os.makedirs(RL_BUFFER_DIR, exist_ok=True)
    out_path = os.path.join(RL_BUFFER_DIR, f"rl_shard_{shard_id:06d}.pt")
    payload = {
        "x": torch.stack([s[0] for s in samples]),
        "pi": torch.stack([s[1] for s in samples]),
        "z": torch.stack([s[2] for s in samples]),
    }
    torch.save(payload, out_path)
    return out_path


def _take_shard_id(counter: "mp.Value", lock: "mp.Lock") -> int:
    with lock:
        sid = int(counter.value)
        counter.value = sid + 1
        return sid


def _selfplay_worker_process(
    worker_id: int,
    num_games: int,
    model_path: str,
    device_str: str,
    simulations: int,
    shard_size: int,
    temperature_moves: int,
    initial_temperature: float,
    mcts_batch_size: int,
    shard_counter: "mp.Value",
    shard_lock: "mp.Lock",
    games_done: "mp.Value",
):
    """Worker process that generates self-play games and writes shards directly to disk."""
    # Limit intra-op thread parallelism per process to avoid CPU oversubscription.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = torch.device(device_str)
    model = ChessNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for _ in range(num_games):
        game_samples = play_single_game(
            model=model,
            device=device,
            simulations=simulations,
            temperature_moves=temperature_moves,
            max_moves=512,
            initial_temperature=initial_temperature,
            mcts_batch_size=mcts_batch_size,
        )

        buffer.extend(game_samples)

        while len(buffer) >= shard_size:
            shard_samples = buffer[:shard_size]
            buffer = buffer[shard_size:]
            sid = _take_shard_id(shard_counter, shard_lock)
            _save_shard(shard_samples, sid)

        with games_done.get_lock():
            games_done.value += 1

    # Flush remainder
    if buffer:
        sid = _take_shard_id(shard_counter, shard_lock)
        _save_shard(buffer, sid)


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
    model: torch.nn.Module | None,
    device: torch.device,
    simulations: int,
    temperature_moves: int = 20,
    initial_temperature: float = 1.25,
    max_moves: int = 512,
    mcts_batch_size: int = 16,
    inference_client=None,
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
            inference_client=inference_client,
            simulations=simulations,
            cpuct=1.5,
            add_dirichlet_noise=True,
            eval_batch_size=int(mcts_batch_size),
        )
        # Align temperature schedule with self-play settings
        mcts.temp_initial = float(initial_temperature)
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


def _batched_selfplay_worker(
    worker_id: int,
    num_games: int,
    request_q,
    response_q,
    out_dir: str,
    simulations: int,
    shard_size: int,
    temperature_moves: int,
    initial_temperature: float,
    max_moves: int,
    mcts_batch_size: int,
    games_counter,
):
    """Top-level entrypoint for spawn-based multiprocessing (Windows safe)."""
    torch.set_num_threads(1)

    client = BatchedInferenceClient(worker_id=int(worker_id), request_q=request_q, response_q=response_q)
    buffer = []

    for _ in range(int(num_games)):
        samples = play_single_game(
            model=None,
            device=torch.device("cpu"),
            inference_client=client,
            simulations=int(simulations),
            temperature_moves=int(temperature_moves),
            initial_temperature=float(initial_temperature),
            max_moves=int(max_moves),
            mcts_batch_size=int(mcts_batch_size),
        )
        buffer.extend(samples)

        while len(buffer) >= int(shard_size):
            shard = buffer[:int(shard_size)]
            buffer = buffer[int(shard_size):]
            _save_shard(out_dir, shard)

        with games_counter.get_lock():
            games_counter.value += 1

    if buffer:
        _save_shard(out_dir, buffer)

def self_play(
    num_games: int,
    simulations: int,
    out_dir: str = str(RL_BUFFER_DIR),
    shard_size: int = 4096,
    temperature_moves: int = 20,
    initial_temperature: float = 1.25,
    max_moves: int = 512,
    workers: int = 1,
    mcts_batch_size: int = 32,
    infer_max_batch: int = 128,
    infer_wait_ms: int = 2,
):
    """
    Multi-worker self-play.

    - If workers == 1: runs in-process, directly calling the model on GPU/CPU.
    - If workers > 1 and CUDA is available: spins up a single GPU inference server that
      batches requests across all workers, while workers run MCTS + game logic on CPU.
    - Each worker writes shards to out_dir; filenames are unique (pid+timestamp).
    """
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(BEST_MODEL_PATH)

    workers = int(max(1, workers))

    if workers == 1 or device.type != "cuda":
        # Single-process path (or CPU-only path): load model locally and run sequentially.
        from src.nn.network import ChessNet
        ckpt = torch.load(model_path, map_location="cpu")
        # Support either raw state_dict or a wrapped dict.
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        model = ChessNet()
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        games_done = 0
        buffer = []
        t0 = time.time()

        for _ in range(int(num_games)):
            samples = play_single_game(
                model=model,
                device=device,
                simulations=int(simulations),
                temperature_moves=int(temperature_moves),
                initial_temperature=float(initial_temperature),
                max_moves=int(max_moves),
                mcts_batch_size=int(mcts_batch_size),
            )
            buffer.extend(samples)
            games_done += 1

            while len(buffer) >= int(shard_size):
                shard = buffer[:int(shard_size)]
                buffer = buffer[int(shard_size):]
                _save_shard(out_dir, shard)

            if games_done % 10 == 0 or games_done == num_games:
                dt = time.time() - t0
                print(f"[Self-play] {games_done}/{num_games} games | {dt:.1f}s elapsed", flush=True)

        if buffer:
            _save_shard(out_dir, buffer)

        return

    # Multi-worker + batched GPU inference server path (CUDA)
    ctx = mp.get_context("spawn")

    request_q = ctx.Queue(maxsize=2048)
    response_qs = [ctx.Queue(maxsize=2048) for _ in range(workers)]

    server = BatchedInferenceServer(
        model_path=model_path,
        device=str(device),
        request_q=request_q,
        response_qs=response_qs,
        max_batch=int(infer_max_batch),
        max_wait_ms=int(infer_wait_ms),
        use_amp=True,
    )
    server.start()

    games_counter = ctx.Value("i", 0)

    # Distribute games across workers
    per = int(num_games) // workers
    rem = int(num_games) % workers

    procs = []
    for wid in range(workers):
        n = per + (1 if wid < rem else 0)
        if n <= 0:
            continue
        p = ctx.Process(
            target=_batched_selfplay_worker,
            args=(
                wid, n,
                request_q, response_qs[wid],
                out_dir,
                int(simulations), int(shard_size),
                int(temperature_moves), float(initial_temperature),
                int(max_moves), int(mcts_batch_size),
                games_counter,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Progress loop
    t0 = time.time()
    last = -1
    while any(p.is_alive() for p in procs):
        with games_counter.get_lock():
            done = int(games_counter.value)
        if done != last:
            dt = time.time() - t0
            print(f"[Self-play] {done}/{num_games} games | {dt:.1f}s elapsed | workers={workers}", flush=True)
            last = done
        time.sleep(1.0)

    for p in procs:
        p.join()

    # Stop server
    try:
        request_q.put(None)
    except Exception:
        pass
    server.join(timeout=5)
