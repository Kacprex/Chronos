import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import io
import time
import random
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
import chess
import chess.pgn
import chess.engine
from tqdm import tqdm

from src.config import AIVSAI, PROJECT_ROOT, LATEST_MODEL_PATH
from src.nn.network import ChessNet
from src.nn.encoding import encode_board


ENGINE_PATH = os.path.join(PROJECT_ROOT, "engine", "stockfish.exe")


def fmt_time(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def stockfish_cp_to_winprob(cp: float) -> float:
    """
    Rough conversion from centipawns to win probability (White POV).
    Heuristic scaling; tune if you want.
    """
    x = cp / 400.0
    return 1.0 / (1.0 + 10.0 ** (-x))


def _read_random_position_from_pgn_text(pgn_text: str, min_ply: int = 6) -> Optional[chess.Board]:
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return None
    if game is None:
        return None

    board = game.board()
    moves = list(game.mainline_moves())
    if len(moves) < min_ply:
        return None

    # pick a ply index not too early
    idx = random.randint(min_ply - 1, len(moves) - 1)
    for mv in moves[:idx]:
        board.push(mv)

    return board.copy(stack=False)


def evaluate_model_vs_stockfish(
    num_positions: int = 500,
    sf_depth: int = 12,
    use_gpu: bool = True,
    model_path: Optional[str] = None,
    out_csv: Optional[str] = None,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Samples random positions from GM PGNs, compares model's value head (mapped to winprob)
    with Stockfish evaluation (mapped to winprob). Returns summary metrics + optional CSV.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    log("Starting Stockfish evaluation...")

    if not os.path.isfile(ENGINE_PATH):
        raise FileNotFoundError(f"Stockfish binary not found at: {ENGINE_PATH}")

    if not os.path.isfile(AIVSAI):
        raise FileNotFoundError(f"AIVSAI_PATH not found: {AIVSAI}")

    df = pd.read_csv(AIVSAI, usecols=["pgn"], low_memory=False)
    log(f"Loaded {len(df)} PGNs from gm_games.csv")

    # Sample positions
    log(f"Sampling {num_positions} positions from GM PGNs...")
    boards: List[chess.Board] = []
    pbar = tqdm(total=num_positions, desc="Sampling positions", dynamic_ncols=True)

    # Avoid infinite loops on bad data:
    max_attempts = max(5000, num_positions * 30)
    attempts = 0

    while len(boards) < num_positions and attempts < max_attempts:
        attempts += 1
        row = df.sample(n=1).iloc[0]
        b = _read_random_position_from_pgn_text(row["pgn"], min_ply=6)
        if b is None:
            continue
        boards.append(b)
        pbar.update(1)

    pbar.close()

    if len(boards) < num_positions:
        log(f"WARNING: only sampled {len(boards)}/{num_positions} positions (data may be sparse).")

    # Init model
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    log(f"Using device for model: {device}")

    model = ChessNet().to(device)

    load_path = model_path or LATEST_MODEL_PATH
    if os.path.isfile(load_path):
        log(f"Loading model weights from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        log(f"WARNING: model file not found at {load_path}; using randomly initialized weights.")

    model.eval()

    # Init Stockfish
    log(f"Starting Stockfish from: {ENGINE_PATH}")
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    model_wps: List[float] = []
    sf_wps: List[float] = []
    rows: List[Dict[str, Any]] = []

    start_time = time.time()

    try:
        for i, board in enumerate(tqdm(boards, desc="Evaluating positions", dynamic_ncols=True), start=1):
            # Model eval
            bt = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0).to(device)

            if device.type == "cuda":
                # New API (fixes deprecation warning)
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    _, v = model(bt)
            else:
                with torch.no_grad():
                    _, v = model(bt)

            v = float(v[0].item())  # assumed in [-1, 1]
            model_wp = (v + 1.0) / 2.0  # to [0,1], White POV

            # Stockfish eval
            info = engine.analyse(board, limit=chess.engine.Limit(depth=sf_depth))
            score = info["score"].white()

            if score.is_mate():
                cp_equiv = 10000.0 if score.mate() and score.mate() > 0 else -10000.0
            else:
                cp_equiv = float(score.cp) if score.cp is not None else 0.0

            sf_wp = stockfish_cp_to_winprob(cp_equiv)

            model_wps.append(model_wp)
            sf_wps.append(sf_wp)

            if out_csv:
                rows.append(
                    {
                        "fen": board.fen(),
                        "model_wp_white": model_wp,
                        "sf_wp_white": sf_wp,
                        "sf_depth": sf_depth,
                        "cp_equiv": cp_equiv,
                    }
                )

            if i % 50 == 0:
                elapsed = time.time() - start_time
                per_pos = elapsed / i
                eta = per_pos * (len(boards) - i)
                log(f"Progress {i}/{len(boards)}, elapsed={fmt_time(elapsed)}, ETA≈{fmt_time(eta)}")

    finally:
        try:
            engine.quit()
        except Exception:
            pass

    mv = torch.tensor(model_wps, dtype=torch.float32)
    sv = torch.tensor(sf_wps, dtype=torch.float32)

    abs_err = torch.mean(torch.abs(mv - sv)).item()

    mv_mean = mv.mean()
    sv_mean = sv.mean()
    cov = torch.mean((mv - mv_mean) * (sv - sv_mean))
    std_mv = mv.std()
    std_sv = sv.std()
    corr = (cov / (std_mv * std_sv + 1e-8)).item()

    total_time = time.time() - start_time
    log("✔ Stockfish evaluation done.")
    log(f"Num positions:      {len(boards)}")
    log(f"Avg abs error (wp): {abs_err:.4f}")
    log(f"Correlation:        {corr:.4f}")
    log(f"Total time:         {fmt_time(total_time)}")

    summary = {
        "num_positions": len(boards),
        "sf_depth": sf_depth,
        "avg_abs_err_wp": abs_err,
        "pearson_corr": corr,
        "total_seconds": total_time,
        "model_path": load_path,
    }

    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log(f"Saved per-position details to: {out_csv}")

    return summary


if __name__ == "__main__":
    # Default run
    evaluate_model_vs_stockfish(num_positions=500, sf_depth=12, use_gpu=True, out_csv=None)
