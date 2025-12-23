import os
import time
import shutil
from datetime import datetime

import chess
import chess.pgn
import torch

from src.config import (
    AIVSAI,
    SFVSAI,
    ENGINE_PATH,
    BEST_MODEL_PATH,
    LATEST_MODEL_PATH,
    RL_RESUME_PATH,
    RESET_LATEST_ON_FAILED_PROMOTION,
    GENERATION_PATH,
    ARCHIVED_MODELS_KEEP,
    MODEL_DIR,
)
from src.nn.network import ChessNet
from src.selfplay.encode_game import GameRecord
from src.selfplay.self_play_worker import self_play
from src.training.train_rl import train_rl
from src.evaluation.promotion import evaluate_and_promote
from src.evaluation.diversity_test import main as run_diversity_test
from src.evaluation.stockfish_eval import evaluate_model_vs_stockfish
from src.logging.discord_logger import log_to_discord
from src.common.generation import read_generation, write_generation

from src.mcts.mcts import MCTS


def log(msg: str, *, discord: bool = True):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)

    # Best-effort: forward *important* events to Discord (rate-limited in the logger).
    # Avoid spamming Discord with loop markers / progress lines.
    if discord:
        noisy_prefixes = (
            "--- Iteration",
            "=== RL LOOP",
            "Iteration ",
            "Self-play ",
            "Train RL ",
            "Training RL",
        )
        if not msg.startswith(noisy_prefixes):
            log_to_discord(line)

def _reset_latest_to_best(*, clear_resume: bool = True, reason: str = "candidate failed promotion"):
    """If a candidate doesn't get promoted, snap latest back to best."""
    try:
        if not os.path.exists(BEST_MODEL_PATH):
            return
        shutil.copy2(BEST_MODEL_PATH, LATEST_MODEL_PATH)
        # The RL resume checkpoint contains optimizer state for the discarded candidate.
        if clear_resume and os.path.exists(RL_RESUME_PATH):
            os.remove(RL_RESUME_PATH)
        log(f"Latest reset to best ({reason}).")
    except Exception as e:
        log(f"WARN: failed to reset latest to best: {e}")


def _ensure_generation_initialized() -> int:
    """Ensure generation.txt exists and seed models/model_0.pth if appropriate."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    gen = read_generation(GENERATION_PATH)
    if not os.path.exists(GENERATION_PATH):
        write_generation(GENERATION_PATH, 0)
        gen = 0

    # Seed rollback archive for generation 0 if best exists and archive missing.
    try:
        model0 = os.path.join(MODEL_DIR, "model_0.pth")
        if os.path.exists(BEST_MODEL_PATH) and not os.path.exists(model0):
            shutil.copy2(BEST_MODEL_PATH, model0)
    except Exception:
        pass

    return int(gen)


def _archive_best_model(gen: int) -> None:
    """Copy current best_model.pth to models/model_{gen}.pth and keep last N."""
    if not os.path.exists(BEST_MODEL_PATH):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    dst = os.path.join(MODEL_DIR, f"model_{int(gen)}.pth")
    try:
        shutil.copy2(BEST_MODEL_PATH, dst)
    except Exception as e:
        log(f"WARN: failed to archive best model to {dst}: {e}")
        return

    keep = int(max(1, ARCHIVED_MODELS_KEEP))
    min_keep_gen = int(gen) - (keep - 1)
    for name in os.listdir(MODEL_DIR):
        if not (name.startswith("model_") and name.endswith(".pth")):
            continue
        stem = name[len("model_") : -len(".pth")]
        if not stem.isdigit():
            continue
        g = int(stem)
        if g < min_keep_gen:
            try:
                os.remove(os.path.join(MODEL_DIR, name))
            except Exception:
                pass


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
        1) self_play -> produces rl shards in RL_BUFFER_DIR
        2) train_rl  -> updates latest model checkpoint(s)
        3) evaluate_and_promote -> may copy latest -> best if threshold met

    Option 8 prompts for the full set of self-play knobs + debug toggle + eval sims.
    """
    import inspect

    log("=== RL LOOP SETUP ===")

    iterations = _read_int("Iterations", 10)

    # Self-play settings
    games_per_iter = _read_int("Self-play games per iteration", 50)
    simulations = _read_int("MCTS simulations per move (self-play)", 200)
    shard_size = _read_int("RL shard size (positions)", 10_000)
    self_play_workers = _read_int("Self-play workers (processes)", 1)
    mcts_batch_size = _read_int("MCTS inference batch size", 32)
    temperature_moves = _read_int("Temperature moves (self-play)", 20)
    initial_temperature = _read_float("Initial temperature (self-play)", 1.25)
    max_moves = _read_int("Max moves per game (self-play)", 512)
    infer_max_batch = _read_int("Inference server max batch", 128)
    infer_wait_ms = _read_int("Inference server max wait (ms)", 2)

    # Debug artifacts toggle
    dbg = input("Debug artifacts (extra PGNs / shard summaries)? [y/N]: ").strip().lower()
    debug = dbg in ("y", "yes", "1", "true")

    # Promotion / evaluation settings
    do_promo = input("Run promotion after each iteration? [Y/n]: ").strip().lower()
    do_promo = (do_promo != "n")

    promo_games = _read_int("Promotion match games", 50) if do_promo else 0
    promo_threshold = _read_float("Promotion threshold (latest winrate)", 0.55) if do_promo else 0.0
    promo_eval_sims = _read_int("MCTS simulations per move (promotion eval)", simulations) if do_promo else 0

    gen = _ensure_generation_initialized()
    log(f"=== RL LOOP START (generation={gen}) ===")

    t0 = time.time()

    for it in range(1, iterations + 1):
        log(f"--- Iteration {it}/{iterations}: self-play (gen={gen}) ---")

        # Call self_play with only the kwargs that exist (keeps compatibility across versions).
        sp_kwargs = dict(
            num_games=games_per_iter,
            simulations=simulations,
            shard_size=shard_size,
            workers=self_play_workers,
            mcts_batch_size=mcts_batch_size,
            temperature_moves=temperature_moves,
            initial_temperature=initial_temperature,
            max_moves=max_moves,
            infer_max_batch=infer_max_batch,
            infer_wait_ms=infer_wait_ms,
            generation=gen,
            loop_iteration=it,
            max_iterations=iterations,
            debug=debug,
        )
        sp_sig = inspect.signature(self_play)
        sp_kwargs = {k: v for k, v in sp_kwargs.items() if k in sp_sig.parameters}
        self_play(**sp_kwargs)

        log(f"--- Iteration {it}/{iterations}: train_rl ---")
        train_rl()

        if do_promo:
            log(f"--- Iteration {it}/{iterations}: evaluate & promote ---")

            promo_kwargs = dict(
                num_games=promo_games,
                threshold=promo_threshold,
                simulations=promo_eval_sims,
                loop_iteration=it,
                max_iterations=iterations,
                selfplay_sims=simulations,
            )
            promo_sig = inspect.signature(evaluate_and_promote)
            promo_kwargs = {k: v for k, v in promo_kwargs.items() if k in promo_sig.parameters}
            promo = evaluate_and_promote(**promo_kwargs)

            # If promoted, bump generation and archive the new best.
            if promo.get("promoted", False):
                gen += 1
                write_generation(GENERATION_PATH, gen)
                _archive_best_model(gen)
                log(f"Promotion succeeded -> generation is now {gen}")

            # Option A: if the candidate fails promotion, reset latest back to best
            # and clear the RL resume checkpoint so training starts clean next iter.
            if RESET_LATEST_ON_FAILED_PROMOTION and not promo.get("promoted", False):
                _reset_latest_to_best(clear_resume=True)

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
            gen = _ensure_generation_initialized()
            games = _read_int("Self-play games", 50)
            sims = _read_int("Simulations", 200)
            shard_size = _read_int("RL shard size (positions)", 10_000)
            workers = _read_int("Self-play workers (processes)", 1)
            mcts_batch = _read_int("MCTS inference batch size", 32)
            self_play(
                num_games=games,
                simulations=sims,
                shard_size=shard_size,
                workers=workers,
                mcts_batch_size=mcts_batch,
                generation=gen,
                loop_iteration=1,
                max_iterations=1,
            )
            train_rl()

        elif choice == "2":
            gen = _ensure_generation_initialized()
            games = _read_int("Self-play games", 50)
            sims = _read_int("Simulations", 200)
            shard_size = _read_int("RL shard size (positions)", 10_000)
            workers = _read_int("Self-play workers (processes)", 1)
            mcts_batch = _read_int("MCTS inference batch size", 32)
            self_play(
                num_games=games,
                simulations=sims,
                shard_size=shard_size,
                workers=workers,
                mcts_batch_size=mcts_batch,
                generation=gen,
                loop_iteration=0,
                max_iterations=0,
            )

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
            gen = _ensure_generation_initialized()
            games = _read_int("Promotion games", 50)
            thr = _read_float("Winrate threshold", 0.55)
            promo = evaluate_and_promote(num_games=games, threshold=thr, loop_iteration=1, max_iterations=1)
            if promo.get("promoted", False):
                gen += 1
                write_generation(GENERATION_PATH, gen)
                _archive_best_model(gen)
                log(f"Promotion succeeded -> generation is now {gen}")

        elif choice == "8":
            run_rl_loop()

        elif choice == "0":
            log("Exiting.")
            break

        else:
            log("Invalid option.")


if __name__ == "__main__":
    main_menu()
