from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import time
from typing import List, Dict, Any

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

from rl.encoding import encode_planes
from rl.util import default_root, append_event, now_ms


def configure_engine(eng: chess.engine.SimpleEngine, *, run_id: str, out_dir: Path, args: argparse.Namespace) -> None:
    cfg = {
        "Hash": int(args.hash_mb),
        "Log": True,
        "LogPath": str(out_dir / "engine_events.jsonl"),
        "RunId": run_id,
        "Hybrid": bool(args.hybrid),
        "Mode": args.mode,
        "AcceptWorseCp": int(args.accept_worse_cp),
        "TopK": int(args.topk),
        "UseNN": bool(args.use_nn),
        "NNModel": args.nn_model or "",
        "NNIntraThreads": int(args.nn_intra),
        "NNInterThreads": int(args.nn_inter),
        "NNPreferCuda": bool(args.nn_cuda),
    }
    # python-chess engine.configure ignores unknown options; that's OK if engine is older.
    eng.configure(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to chronos_engine executable")
    ap.add_argument("--run", default="", help="Run id (folder). Default: rl_<timestamp>")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--movetime-ms", type=int, default=100)
    ap.add_argument("--depth", type=int, default=0, help="If >0, use depth instead of movetime")
    ap.add_argument("--hash-mb", type=int, default=256)

    # Hybrid knobs
    ap.add_argument("--hybrid", action="store_true", default=True)
    ap.add_argument("--mode", default="blitz", choices=["classic", "blitz"])
    ap.add_argument("--accept-worse-cp", type=int, default=40)
    ap.add_argument("--topk", type=int, default=8)

    # NN
    ap.add_argument("--use-nn", action="store_true", default=False)
    ap.add_argument("--nn-model", default="", help="Path to ONNX model (requires engine built with ONNX Runtime)")
    ap.add_argument("--nn-intra", type=int, default=1)
    ap.add_argument("--nn-inter", type=int, default=1)
    ap.add_argument("--nn-cuda", action="store_true", default=False)

    args = ap.parse_args()

    root = default_root()
    run_id = args.run or f"rl_{int(time())}"
    out_dir = root / "shards" / "rl" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = out_dir / "selfplay_raw.jsonl"
    pgn_path = out_dir / "selfplay.pgn"

    append_event(root, {"ts_ms": now_ms(), "type": "rl_selfplay_start", "run_id": run_id, "games": args.games, "out_dir": str(out_dir)})

    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        configure_engine(eng, run_id=run_id, out_dir=out_dir, args=args)

        with raw_jsonl.open("w", encoding="utf-8") as out, pgn_path.open("w", encoding="utf-8") as pgnf:
            for game_id in tqdm(range(1, args.games + 1), desc="Selfplay games"):
                board = chess.Board()

                # PGN game
                game = chess.pgn.Game()
                game.headers["Event"] = "Chronos RL Selfplay"
                game.headers["Site"] = "local"
                game.headers["Date"] = ""
                game.headers["Round"] = str(game_id)
                game.headers["White"] = "Chronos"
                game.headers["Black"] = "Chronos"
                node = game

                records: List[Dict[str, Any]] = []
                ply = 0

                while ply < args.max_plies and not board.is_game_over(claim_draw=True):
                    planes = encode_planes(board)
                    limit = chess.engine.Limit(depth=args.depth) if args.depth > 0 else chess.engine.Limit(time=max(0.01, args.movetime_ms / 1000.0))
                    res = eng.play(board, limit)

                    if res.move is None:
                        break

                    rec = {
                        "run_id": run_id,
                        "game_id": game_id,
                        "ply": ply,
                        "fen": board.fen(),
                        "planes": planes,
                        "move": res.move.uci(),
                        "meta": {
                            "movetime_ms": args.movetime_ms,
                            "depth": args.depth,
                            "mode": args.mode,
                        },
                    }
                    records.append(rec)

                    board.push(res.move)
                    node = node.add_variation(res.move)
                    ply += 1

                result = board.result(claim_draw=True)
                game.headers["Result"] = result
                pgnf.write(str(game) + "\n\n")

                for rec in records:
                    rec["result"] = result
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        eng.quit()

    append_event(root, {"ts_ms": now_ms(), "type": "rl_selfplay_done", "run_id": run_id, "raw": str(raw_jsonl), "pgn": str(pgn_path)})
    print(f"Done. Raw: {raw_jsonl}")
    print(f"PGN: {pgn_path}")


if __name__ == "__main__":
    main()
