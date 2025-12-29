from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple

import chess
import chess.engine
from tqdm import tqdm

from rl.util import default_root, append_event, now_ms


def play_game(engine_path: str, cand_onnx: str, best_onnx: str, movetime_ms: int, cand_is_white: bool) -> str:
    # returns result "1-0", "0-1", "1/2-1/2" from white perspective (white engine is candidate or best)
    e_white = chess.engine.SimpleEngine.popen_uci(engine_path)
    e_black = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        if cand_is_white:
            e_white.configure({"UseNN": True, "NNModel": cand_onnx, "RunId": "promo_cand"})
            e_black.configure({"UseNN": True, "NNModel": best_onnx, "RunId": "promo_best"})
        else:
            e_white.configure({"UseNN": True, "NNModel": best_onnx, "RunId": "promo_best"})
            e_black.configure({"UseNN": True, "NNModel": cand_onnx, "RunId": "promo_cand"})

        board = chess.Board()
        limit = chess.engine.Limit(time=max(0.01, movetime_ms / 1000.0))

        while not board.is_game_over(claim_draw=True):
            if board.turn == chess.WHITE:
                mv = e_white.play(board, limit).move
            else:
                mv = e_black.play(board, limit).move
            if mv is None:
                break
            board.push(mv)

        res = board.result(claim_draw=True)
        return res
    finally:
        e_white.quit()
        e_black.quit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="chronos_engine path")
    ap.add_argument("--candidate", required=True, help="candidate ONNX")
    ap.add_argument("--best", required=True, help="best ONNX")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime-ms", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--promote-to", default="", help="Override destination ONNX. Default: E:/chronos/models/best.onnx")
    args = ap.parse_args()

    root = default_root()
    run_id = f"promo_{int(time())}"

    cand = Path(args.candidate)
    best = Path(args.best)
    dest = Path(args.promote_to) if args.promote_to else (root / "models" / "best.onnx")
    dest.parent.mkdir(parents=True, exist_ok=True)

    append_event(root, {"ts_ms": now_ms(), "type": "rl_promo_start", "run_id": run_id, "games": args.games, "candidate": str(cand), "best": str(best)})

    cand_wins = 0
    draws = 0

    for i in tqdm(range(args.games), desc="Promo games"):
        cand_is_white = (i % 2 == 0)
        res = play_game(args.engine, str(cand), str(best), args.movetime_ms, cand_is_white=cand_is_white)
        if res == "1/2-1/2":
            draws += 1
        else:
            white_won = (res == "1-0")
            cand_won = (white_won and cand_is_white) or ((not white_won) and (not cand_is_white))
            if cand_won:
                cand_wins += 1

    win_rate = cand_wins / max(1, (args.games - draws)) if (args.games - draws) > 0 else 0.0

    promoted = False
    if win_rate >= args.threshold:
        shutil.copy2(cand, dest)
        promoted = True
        promo_log = root / "logs" / "promotions.txt"
        promo_log.parent.mkdir(parents=True, exist_ok=True)
        with promo_log.open("a", encoding="utf-8") as f:
            f.write(f"{now_ms()} PROMOTE {cand} -> {dest} win_rate={win_rate:.3f} draws={draws}/{args.games}\\n")

    append_event(root, {"ts_ms": now_ms(), "type": "rl_promo_done", "run_id": run_id, "win_rate": win_rate, "cand_wins": cand_wins, "draws": draws, "promoted": promoted, "dest": str(dest)})
    print(f"Candidate wins: {cand_wins}, draws: {draws}, games: {args.games}")
    print(f"Win rate (excluding draws): {win_rate:.3f}")
    print("PROMOTED" if promoted else "not promoted")


if __name__ == "__main__":
    main()
