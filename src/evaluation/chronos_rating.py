import os
import time
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import chess
import chess.engine

from src.config import ENGINE_PATH
from src.nn.network import ChessNet
from src.mcts.mcts import MCTS


@dataclass
class RatingResult:
    score: float           # mean score from Chronos perspective (1=win, 0.5=draw, 0=loss)
    elo_diff: float        # estimated Elo difference vs this SF config
    rating_index: float    # arbitrary anchored index (1500 + elo_diff)
    depth: int
    num_games: int
    seconds: float


def _score_to_elo_diff(score: float) -> float:
    # Convert expected score to Elo difference: E = 1/(1+10^(-d/400)) => d = 400*log10(E/(1-E))
    eps = 1e-6
    s = float(np.clip(score, eps, 1.0 - eps))
    return 400.0 * math.log10(s / (1.0 - s))


def _play_one_game_vs_stockfish(
    model: ChessNet,
    device: torch.device,
    *,
    depth: int,
    simulations: int,
    max_moves: int = 512,
    chronos_is_white: bool = True,
) -> float:
    board = chess.Board()

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    try:
        mcts = MCTS(model=model, device=device, simulations=simulations, cpuct=1.5, add_dirichlet_noise=False)

        # deterministic play for rating
        if hasattr(mcts, "temp_moves"):
            mcts.temp_moves = 0
        if hasattr(mcts, "temp_initial"):
            mcts.temp_initial = 1.0

        for _ in range(max_moves):
            if board.is_game_over(claim_draw=True):
                break

            chronos_turn = (board.turn == chess.WHITE) if chronos_is_white else (board.turn == chess.BLACK)

            if chronos_turn:
                moves, probs = mcts.run(board, move_number=1, add_noise=False)
                if not moves:
                    break
                mv = moves[int(np.argmax(probs))]
            else:
                mv = engine.play(board, chess.engine.Limit(depth=depth)).move

            board.push(mv)

        result = board.result(claim_draw=True)
    finally:
        try:
            engine.quit()
        except Exception:
            pass

    # score from Chronos perspective
    if result == "1-0":
        return 1.0 if chronos_is_white else 0.0
    if result == "0-1":
        return 0.0 if chronos_is_white else 1.0
    return 0.5


def estimate_chronos_rating_vs_stockfish(
    model_path: str,
    *,
    depth: int = 12,
    num_games: int = 20,
    simulations: int = 200,
    seed: int = 123,
) -> RatingResult:
    """
    Estimate a *relative* rating against your local Stockfish configuration.

    Returns:
      - score: average points (win=1, draw=0.5, loss=0)
      - elo_diff: estimated Elo difference vs Stockfish at the given depth (relative)
      - rating_index: 1500 + elo_diff (a convenient single number; NOT a real chess.com Elo)
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores = []
    for g in range(1, num_games + 1):
        chronos_is_white = (g % 2 == 1)
        scores.append(
            _play_one_game_vs_stockfish(
                model,
                device,
                depth=depth,
                simulations=simulations,
                chronos_is_white=chronos_is_white,
            )
        )

    score = float(np.mean(scores)) if scores else 0.5
    elo_diff = _score_to_elo_diff(score)
    rating_index = 1500.0 + elo_diff
    return RatingResult(
        score=score,
        elo_diff=elo_diff,
        rating_index=rating_index,
        depth=depth,
        num_games=num_games,
        seconds=time.time() - t0,
    )
