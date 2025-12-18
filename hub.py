import sys, os
sys.path.append(os.path.abspath("."))

import numpy as np
import time
from datetime import datetime

from src.training.train_supervised import train as run_supervised
from src.training.train_rl import train_rl
from src.selfplay.self_play_worker import self_play
from src.selfplay.encode_game import GameRecord, build_game_record_from_moves
from src.evaluation.diversity_test import main as run_diversity_test
from src.evaluation.promotion import evaluate_and_promote
from src.evaluation.stockfish_eval import evaluate_model_vs_stockfish
from src.config import AIVSAI, SFVSAI

import chess
import chess.engine
import torch

from src.nn.network import ChessNet
from src.nn.encoding import encode_board, move_to_index
from src.config import (
    BEST_MODEL_PATH,
    LATEST_MODEL_PATH,
    ENGINE_PATH,
)


def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


# ------------------------------------------------------------
# PLAY AI vs AI
# ------------------------------------------------------------
def play_ai_vs_ai(simulations=200, max_moves=200, save_pgn=True):
    from src.mcts.mcts import MCTS

    log("Loading best model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_w = ChessNet().to(device)
    model_b = ChessNet().to(device)

    model_w.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))
    model_b.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))

    model_w.eval()
    model_b.eval()

    mcts_w = MCTS(model_w, device=device, simulations=simulations)
    mcts_b = MCTS(model_b, device=device, simulations=simulations)

    board = chess.Board()
    moves_uci = []

    log("Starting AI vs AI match...")

    while not board.is_game_over(claim_draw=True) and len(moves_uci) < max_moves:
        if board.turn == chess.WHITE:
           _, pi = mcts_w.run(board, move_number=len(board.move_stack), add_noise=False)
        else:
            _, pi = mcts_w.run(board, move_number=len(board.move_stack), add_noise=False)

        move_idx = int(np.argmax(pi))
        legal = list(board.legal_moves)

        chosen = None
        for mv in legal:
            if move_to_index(mv) == move_idx:
                chosen = mv
                break
        if chosen is None:
            chosen = legal[0]

        moves_uci.append(chosen.uci())
        board.push(chosen)

    result = board.result(claim_draw=True)

    log(f"Game finished: {result}")

    if save_pgn:
        record = GameRecord(moves_uci=moves_uci, result=result)
        record.save_pgn(AIVSAI)
        log("PGN saved to ai_vs_ai_games.pgn")

    return result


# ------------------------------------------------------------
# PLAY Stockfish vs AI
# ------------------------------------------------------------
def play_sf_vs_ai(sf_depth=12, simulations=200, save_pgn=True):
    from src.mcts.mcts import MCTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("Loading AI model...")
    model = ChessNet().to(device)
    model.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))
    model.eval()

    mcts = MCTS(model=model, device=device, simulations=simulations)

    log("Starting Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    board = chess.Board()
    moves_uci = []

    log("Stockfish (White) vs AI (Black)...")

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            info = engine.analyse(board, limit=chess.engine.Limit(depth=sf_depth))
            move = info["pv"][0]
        else:
            _, pi = mcts.run(board, move_number=len(board.move_stack), add_noise=False)
            move_idx = int(np.argmax(pi))

            legal = list(board.legal_moves)
            chosen = None
            for mv in legal:
                if move_to_index(mv) == move_idx:
                    chosen = mv
                    break
                
            if chosen is None:
                chosen = legal[0]

            move = chosen


        moves_uci.append(move.uci())
        board.push(move)

    engine.quit()

    result = board.result()
    log(f"Game finished: {result}")

    if save_pgn:
        record = GameRecord(moves_uci=moves_uci, result=result, white_name="Stockfish", black_name="AI")
        record.save_pgn(SFVSAI)
        log("PGN saved to sf_vs_ai_games.pgn")

    return result


# ------------------------------------------------------------
# MAIN HUB MENU
# ------------------------------------------------------------
def main_menu():
    while True:
        print("\n=================== AI CHESS HUB ===================")
        print("1. Run Self-Play → RL Training Loop (manual)")
        print("2. Generate Self-Play Games Only")
        print("3. Play AI vs AI and export PGN")
        print("4. Play Stockfish vs AI and export PGN")
        print("5. Run diveristy test")
        print("6. Run Stockfish Evaluation on Model")
        print("7. Evaluate Latest vs Best (Promotion)")
        print("8. Exit")
        print("=====================================================")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            num_games = int(input("Self-play games to generate: "))
            sims = int(input("MCTS simulations per move: "))
            self_play(num_games=num_games, simulations=sims)
            train_rl()
            print("RL cycle complete!")

        elif choice == "2":
            num_games = int(input("Number of self-play games: "))
            sims = int(input("MCTS simulations per move: "))
            self_play(num_games=num_games, simulations=sims)

        elif choice == "3":
            num_games = int(input("Number of AI vs AI games: "))
            sims = int(input("MCTS simulations per move: "))
            for i in range(num_games):
                log(f"AI vs AI game {i+1}/{num_games}")
                play_ai_vs_ai(simulations=sims)

        elif choice == "4":
            num_games = int(input("Number of Stockfish vs AI games: "))
            depth = int(input("Stockfish depth: "))
            sims = int(input("AI MCTS simulations per move: "))
            for i in range(num_games):
                log(f"Stockfish vs AI game {i+1}/{num_games}")
                play_sf_vs_ai(sf_depth=depth, simulations=sims)
        elif choice =="5":
            run_diversity_test()
        elif choice == "6":
            positions = int(input("Number of positions to evaluate: "))
            depth = int(input("SF depth: "))
            evaluate_model_vs_stockfish(num_positions=positions, sf_depth=depth)

        elif choice == "7":
            games = int(input("Games for evaluation: "))
            evaluate_and_promote(num_games=games)

        elif choice == "8":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
