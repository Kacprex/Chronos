import os
import io
import time
import random
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
import chess
import chess.pgn
import chess.engine
from tqdm import tqdm

from src.config import ENGINE_PATH, GM_GAMES_PATH, BEST_MODEL_PATH, LATEST_MODEL_PATH
from src.nn.network import ChessNet
from src.nn.encoding import encode_board


from src.common.log import log


def _cp_to_value(cp: Optional[int], mate: Optional[int]) -> float:
    """Convert Stockfish score to [-1, 1] value. Uses tanh on centipawns, hard sign on mate."""
    if mate is not None:
        # mate > 0 means side-to-move mates; mate < 0 means side-to-move is mated
        return 1.0 if mate > 0 else -1.0
    if cp is None:
        return 0.0
    # Typical scaling: 400cp ~ tanh(1) ~ 0.76
    return float(np.tanh(cp / 400.0))


def _read_random_position_from_pgn_text(pgn_text: str, min_ply: int = 6, max_ply: int = 160) -> Optional[chess.Board]:
    """Parse PGN string and return a board from a random ply in [min_ply, max_ply]."""
    if not pgn_text or not isinstance(pgn_text, str):
        return None

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return None
    if game is None:
        return None

    board = game.board()
    moves = list(game.mainline_moves())
    if len(moves) < min_ply + 1:
        return None

    hi = min(len(moves), max_ply)
    ply = random.randint(min_ply, max(min_ply, hi - 1))

    for mv in moves[:ply]:
        if mv not in board.legal_moves:
            return None
        board.push(mv)

    if board.is_game_over(claim_draw=True):
        return None

    return board


def _reservoir_sample_pgn_rows(csv_path: str, k: int, seed: int = 42, chunksize: int = 50_000) -> List[str]:
    """Reservoir-sample K PGN strings from a large CSV without loading it all into memory."""
    random.seed(seed)
    reservoir: List[str] = []
    seen = 0

    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c.strip().lower() == "pgn",
        chunksize=chunksize,
        low_memory=False,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
        encoding="utf-8",
    )

    for chunk in reader:
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        for pgn_text in chunk["pgn"].tolist():
            seen += 1
            if len(reservoir) < k:
                reservoir.append(pgn_text)
            else:
                j = random.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = pgn_text

    return reservoir


def evaluate_model_vs_stockfish(
    num_positions: int = 200,
    sf_depth: int = 12,
    csv_path: str = GM_GAMES_PATH,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    seed: int = 42,
) -> None:
    """
    Evaluate the network **value head** against Stockfish on random sampled positions.

    - Positions are sampled from the CSV `pgn` column.
    - Stockfish evaluates each position at `sf_depth`.
    - Model predicts value in [-1, 1].
    - Prints MAE/MSE/correlation + a few sample rows.

    This is intended as a **sanity check**, not a perfect benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_path is None:
        model_path = LATEST_MODEL_PATH if os.path.exists(LATEST_MODEL_PATH) else BEST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(f"Stockfish not found: {ENGINE_PATH}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log(f"Model: {model_path}")
    log(f"Stockfish: {ENGINE_PATH} (depth={sf_depth})")
    log(f"CSV source: {csv_path}")
    log(f"Sampling {num_positions} positions...")

    # Sample PGNs then extract random positions
    pgn_rows = _reservoir_sample_pgn_rows(csv_path, k=max(num_positions * 3, 500), seed=seed)
    boards: List[chess.Board] = []

    for pgn_text in pgn_rows:
        if len(boards) >= num_positions:
            break
        b = _read_random_position_from_pgn_text(pgn_text, min_ply=6)
        if b is not None:
            boards.append(b)

    if len(boards) < num_positions:
        log(f"WARNING: only collected {len(boards)}/{num_positions} positions (PGNs may be malformed/sparse).")
        num_positions = len(boards)

    # Load model
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    log(f"Device: {device}")

    model = ChessNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Evaluate
    sf_values: List[float] = []
    nn_values: List[float] = []

    t0 = time.time()
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    try:
        for b in tqdm(boards, desc="Evaluating"):
            # Stockfish score from side-to-move POV
            info = engine.analyse(b, chess.engine.Limit(depth=sf_depth))
            score = info.get("score")
            cp = None
            mate = None
            if score is not None:
                pov = score.pov(b.turn)
                mate = pov.mate()
                cp = pov.score(mate_score=10_000)
                # If mate is available, cp can be None; we handle that in converter
                if mate is not None:
                    cp = None

            sf_v = _cp_to_value(cp, mate)
            sf_values.append(sf_v)

            x = torch.from_numpy(encode_board(b)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                _policy, v = model(x)
            nn_values.append(float(v.squeeze().cpu().item()))
    finally:
        engine.quit()

    elapsed = time.time() - t0

    sf_arr = np.array(sf_values, dtype=np.float32)
    nn_arr = np.array(nn_values, dtype=np.float32)

    mae = float(np.mean(np.abs(sf_arr - nn_arr)))
    mse = float(np.mean((sf_arr - nn_arr) ** 2))
    corr = float(np.corrcoef(sf_arr, nn_arr)[0, 1]) if len(sf_arr) >= 2 else float("nan")

    log(f"Done in {elapsed:.1f}s  ({(len(boards) / elapsed) if elapsed > 0 else 0.0:.2f} pos/s)")
    log(f"MAE={mae:.4f}  MSE={mse:.4f}  Corr={corr:.4f}")

    # Print a small sample table
    show_n = min(8, len(boards))
    log("Sample predictions (SF vs NN):")
    for i in range(show_n):
        log(f"  {i+1:02d}) sf={sf_arr[i]:+.3f}  nn={nn_arr[i]:+.3f}  delta={(nn_arr[i]-sf_arr[i]):+.3f}  fen={boards[i].fen()}")


if __name__ == "__main__":
    evaluate_model_vs_stockfish(num_positions=200, sf_depth=12)
