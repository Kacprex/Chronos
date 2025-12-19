import os
import time
from datetime import datetime

import chess
import chess.pgn
import torch

from src.config import AIVSAI, SFVSAI, ENGINE_PATH, BEST_MODEL_PATH
from src.nn.network import ChessNet
from src.selfplay.encode_game import GameRecord
from src.selfplay.self_play_worker import self_play
from src.training.train_rl import train_rl
from src.evaluation.promotion import evaluate_and_promote
from src.evaluation.diversity_test import main as run_diversity_test
from src.evaluation.stockfish_eval import evaluate_model_vs_stockfish

from src.mcts.mcts import MCTS


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def _read_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        log("Invalid integer; using default.")
        return default


def _read_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        log("Invalid number; using default.")
        return default


def play_ai_vs_ai(simulations: int = 200, max_moves: int = 200, save_pgn: bool = True):
    """
    Plays a single AI vs AI game using BEST_MODEL_PATH for both sides and saves PGN to AIVSAI.
    Uses MCTS.run() returning (moves, probs).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_w = ChessNet().to(device)
    model_b = ChessNet().to(device)

    if os.path.exists(BEST_MODEL_PATH):
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
        model_w.load_state_dict(ckpt)
        model_b.load_state_dict(ckpt)
    else:
        log(f"BEST_MODEL_PATH not found: {BEST_MODEL_PATH}")
        return

    model_w.eval()
    model_b.eval()

    mcts_w = MCTS(model_w, device, simulations=simulations, cpuct=1.5, add_dirichlet_noise=False)
    mcts_b = MCTS(model_b, device, simulations=simulations, cpuct=1.5, add_dirichlet_noise=False)

    board = chess.Board()
    moves_uci = []

    for ply in range(1, max_moves + 1):
        if board.is_game_over(claim_draw=True):
            break

        mcts = mcts_w if board.turn == chess.WHITE else mcts_b
        moves, probs = mcts.run(board, move_number=ply, add_noise=False)
        if not moves:
            break

        # deterministic best move
        best_idx = int(probs.argmax())
        move = moves[best_idx]

        moves_uci.append(move.uci())
        board.push(move)

    result = board.result(claim_draw=True)

    if save_pgn:
        rec = GameRecord(
            moves_uci=moves_uci,
            result=result,
            white_name="Chronos",
            black_name="Chronos",
            event="AI vs AI",
            site="local",
            date=datetime.now().strftime("%Y.%m.%d"),
            mcts_sims=str(simulations),
        )
        rec.save_pgn(AIVSAI, extra_headers={"Result": result})
        log(f"Saved PGN to: {AIVSAI}")

    log(f"AI vs AI finished. Result: {result}")


def play_sf_vs_ai(sf_depth: int = 12, simulations: int = 200, max_moves: int = 200, save_pgn: bool = True):
    """
    Plays Stockfish (Black) vs AI (White) and saves PGN to SFVSAI.
    """
    import chess.engine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)

    if os.path.exists(BEST_MODEL_PATH):
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt)
    else:
        log(f"BEST_MODEL_PATH not found: {BEST_MODEL_PATH}")
        return

    model.eval()
    mcts = MCTS(model, device, simulations=simulations, cpuct=1.5, add_dirichlet_noise=False)

    board = chess.Board()
    moves_uci = []

    if not os.path.exists(ENGINE_PATH):
        log(f"Stockfish not found at: {ENGINE_PATH}")
        return

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    try:
        for ply in range(1, max_moves + 1):
            if board.is_game_over(claim_draw=True):
                break

            if board.turn == chess.WHITE:
                # AI move
                moves, probs = mcts.run(board, move_number=ply, add_noise=False)
                if not moves:
                    break
                best_idx = int(probs.argmax())
                move = moves[best_idx]
            else:
                # Stockfish move
                info = engine.play(board, chess.engine.Limit(depth=sf_depth))
                move = info.move

            moves_uci.append(move.uci())
            board.push(move)

        result = board.result(claim_draw=True)
    finally:
        engine.quit()

    if save_pgn:
        rec = GameRecord(
            moves_uci=moves_uci,
            result=result,
            white_name="Chronos",
            black_name="Stockfish",
            event="SF vs AI",
            site="local",
            date=datetime.now().strftime("%Y.%m.%d"),
            mcts_sims=str(simulations),
            sf_depth=str(sf_depth),
        )
        rec.save_pgn(SFVSAI, extra_headers={"Result": result})
        log(f"Saved PGN to: {SFVSAI}")

    log(f"SF vs AI finished. Result: {result}")


def run_rl_loop():
    """
    RL loop:
      for i in iterations:
        1) self_play -> produces rl_shard_*.pt in RL_BUFFER_DIR
        2) train_rl  -> updates latest model checkpoint(s)
        3) evaluate_and_promote -> may copy latest -> best if threshold met
    """
    log("=== RL LOOP SETUP ===")
    iterations = _read_int("Iterations", 10)
    games_per_iter = _read_int("Self-play games per iteration", 50)
    simulations = _read_int("MCTS simulations per move (self-play)", 200)
    shard_size = _read_int("RL shard size (positions)", 10_000)

    do_promo = input("Run promotion after each iteration? [Y/n]: ").strip().lower()
    do_promo = (do_promo != "n")

    promo_games = _read_int("Promotion match games", 50) if do_promo else 0
    promo_sims = _read_int("Promotion MCTS simulations per move", 200) if do_promo else 200
    promo_threshold = _read_float("Promotion threshold (latest winrate)", 0.55) if do_promo else 0.0

    log("=== RL LOOP START ===")
    t0 = time.time()

    for it in range(1, iterations + 1):
        log(f"--- Iteration {it}/{iterations}: self-play ---")
        self_play(num_games=games_per_iter, simulations=simulations, shard_size=shard_size)

        log(f"--- Iteration {it}/{iterations}: train_rl ---")
        train_rl()

        if do_promo:
            log(f"--- Iteration {it}/{iterations}: evaluate & promote ---")
            evaluate_and_promote(num_games=promo_games, threshold=promo_threshold, simulations=promo_sims, loop_iteration=it, max_iterations=iterations, selfplay_sims=simulations)

        elapsed = time.time() - t0
        log(f"Iteration {it} complete. Total elapsed: {elapsed:.1f}s")

    log("=== RL LOOP DONE ===")


def main_menu():
    while True:
        print("\n=== Chronos Hub ===")
        print("1) Self-play + Train RL (single cycle)")
        print("2) Self-play only")
        print("3) AI vs AI (save PGN)")
        print("4) Stockfish vs AI (save PGN)")
        print("5) Diversity test (analyze PGNs)")
        print("6) Stockfish eval (value head)")
        print("7) Evaluate & Promote (latest vs best)")
        print("8) Run RL loop (self-play → train_rl → promote)")
        print("0) Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            games = _read_int("Self-play games", 50)
            sims = _read_int("Simulations", 200)
            shard_size = _read_int("RL shard size (positions)", 10_000)
            self_play(num_games=games, simulations=sims, shard_size=shard_size)
            train_rl()

        elif choice == "2":
            games = _read_int("Self-play games", 50)
            sims = _read_int("Simulations", 200)
            shard_size = _read_int("RL shard size (positions)", 10_000)
            self_play(num_games=games, simulations=sims, shard_size=shard_size)

        elif choice == "3":
            sims = _read_int("Simulations", 200)
            play_ai_vs_ai(simulations=sims)

        elif choice == "4":
            sf_depth = _read_int("Stockfish depth", 12)
            sims = _read_int("Simulations", 200)
            play_sf_vs_ai(sf_depth=sf_depth, simulations=sims)

        elif choice == "5":
            run_diversity_test()

        elif choice == "6":
            npos = _read_int("Number of positions", 200)
            sf_depth = _read_int("Stockfish depth", 12)
            evaluate_model_vs_stockfish(num_positions=npos, sf_depth=sf_depth)

        elif choice == "7":
            games = _read_int("Promotion games", 50)
            thr = _read_float("Winrate threshold", 0.55)
            sims = _read_int("Promotion MCTS simulations per move", 200)
            evaluate_and_promote(num_games=games, threshold=thr, simulations=sims, loop_iteration=1, max_iterations=1)

        elif choice == "8":
            run_rl_loop()

        elif choice == "0":
            log("Exiting.")
            break

        else:
            log("Invalid option.")


if __name__ == "__main__":
    main_menu()
