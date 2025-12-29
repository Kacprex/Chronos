from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, Any, List, Tuple

import chess
import chess.engine
from tqdm import tqdm

from rl.encoding import encode_planes
from nn.move_index import move_to_index, MOVE_SPACE
from rl.util import default_root, append_event, now_ms


@dataclass
class Config:
    raw_jsonl: Path
    stockfish: Path
    out_jsonl: Path
    depth: int = 12
    movetime_ms: int = 0
    mistake_cp: int = 70
    horizon: int = 4
    multipv: int = 0
    policy_temp: float = 150.0


def pov_cp(score: chess.engine.PovScore, turn: bool) -> int:
    # score is pov(WHITE); convert to perspective of side-to-move
    sc = score.pov(chess.WHITE)
    cp = sc.score(mate_score=100000)
    if cp is None:
        # mate: very large
        cp = 100000 if sc.mate() and sc.mate() > 0 else -100000
    # convert to side-to-move perspective
    return cp if turn == chess.WHITE else -cp


def tanh_value(cp: int, scale: float = 600.0) -> float:
    x = max(-3000.0, min(3000.0, float(cp)))
    return float(math.tanh(x / scale))


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def eval_board(sf: chess.engine.SimpleEngine, board: chess.Board, cfg: Config) -> int:
    limit = chess.engine.Limit(depth=cfg.depth) if cfg.movetime_ms <= 0 else chess.engine.Limit(time=max(0.01, cfg.movetime_ms / 1000.0))
    info = sf.analyse(board, limit)
    score = info["score"]
    return pov_cp(score, board.turn)



def softmax_cps(cps: list[int], temp: float) -> list[float]:
    if not cps:
        return []
    mx = max(cps)
    exps = [pow(2.718281828, (cp - mx) / max(1e-6, temp)) for cp in cps]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(cps)] * len(cps)
    return [e / s for e in exps]


def eval_board_multipv(sf: chess.engine.SimpleEngine, board: chess.Board, cfg: Config) -> list[tuple[chess.Move, int]]:
    '''
    Returns list of (move, cp_from_side_to_move_pov) for MultiPV lines.
    '''
    if cfg.multipv <= 0:
        return []
    limit = chess.engine.Limit(depth=cfg.depth) if cfg.movetime_ms <= 0 else chess.engine.Limit(time=max(0.01, cfg.movetime_ms / 1000.0))
    infos = sf.analyse(board, limit, multipv=int(cfg.multipv))
    out: list[tuple[chess.Move, int]] = []
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        mv = pv[0]
        cp = pov_cp(info["score"], board.turn)
        out.append((mv, cp))
    return out


def load_records(path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            recs.append(json.loads(ln))
    recs.sort(key=lambda r: (r["game_id"], r["ply"]))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="selfplay_raw.jsonl")
    ap.add_argument("--stockfish", required=True, help="Path to stockfish executable")
    ap.add_argument("--out", default="", help="Output labeled JSONL. Default: sibling labeled.jsonl")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--movetime-ms", type=int, default=0)
    ap.add_argument("--mistake-cp", type=int, default=70)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--multipv", type=int, default=0, help="If >0, also write sparse policy targets from Stockfish MultiPV")
    ap.add_argument("--policy-temp", type=float, default=150.0, help="Softmax temperature in centipawns")
    args = ap.parse_args()

    root = default_root()
    raw = Path(args.raw)
    out = Path(args.out) if args.out else raw.parent / "labeled.jsonl"
    cfg = Config(
        raw_jsonl=raw,
        stockfish=Path(args.stockfish),
        out_jsonl=out,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        mistake_cp=args.mistake_cp,
        horizon=args.horizon,
        multipv=args.multipv,
        policy_temp=float(args.policy_temp),
    )

    run_id = "rl"
    try:
        # infer run_id from first line if present
        with raw.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
            if first:
                run_id = json.loads(first).get("run_id", "rl")
    except Exception:
        pass

    append_event(root, {"ts_ms": now_ms(), "type": "rl_label_start", "run_id": run_id, "raw": str(raw), "out": str(out), "depth": cfg.depth})

    recs = load_records(cfg.raw_jsonl)

    # Group by game
    games: Dict[int, List[Dict[str, Any]]] = {}
    for r in recs:
        games.setdefault(int(r["game_id"]), []).append(r)

    sf = chess.engine.SimpleEngine.popen_uci(str(cfg.stockfish))
    try:
        labeled_rows: List[Dict[str, Any]] = []

        for gid, g in tqdm(games.items(), desc="Label games"):
            # Compute eval before and after each move
            cp_before: List[int] = []
            cp_after: List[int] = []
            mover_is_white: List[bool] = []
            mistakes: List[int] = []

            for rec in g:
                board = chess.Board(rec["fen"])
                mover_is_white.append(board.turn == chess.WHITE)

                cp0 = eval_board(sf, board, cfg)
                cp_before.append(cp0)

                mv = chess.Move.from_uci(rec["move"])
                if mv not in board.legal_moves:
                    # invalid (shouldn't happen), treat as big mistake
                    cp_after.append(cp0 - 500)
                    mistakes.append(1)
                    continue

                board.push(mv)
                cp1 = eval_board(sf, board, cfg)
                cp_after.append(cp1)

                delta = cp1 - cp0  # from mover perspective because pov follows board.turn at each eval call
                # After push, board.turn flipped, so cp1 is from opponent-to-move perspective.
                # Convert cp1 to mover perspective by negating.
                cp1_mover = -cp1
                delta_mover = cp1_mover - cp0

                mistakes.append(1 if delta_mover <= -cfg.mistake_cp else 0)

            # Precompute "pressure soon" for each ply: opponent mistake within horizon moves
            n = len(g)
            opp_mistake_soon: List[float] = [0.0] * n
            volatility: List[float] = [0.0] * n

            for i in range(n):
                mover_white = mover_is_white[i]
                # look ahead at opponent moves after our move (i+1, i+3, ...)
                horizon_end = min(n, i + 1 + cfg.horizon)
                hit = 0
                # Also compute volatility as max abs swing in cp_before within horizon window
                base = cp_before[i]
                vmax = 0.0
                for j in range(i + 1, horizon_end):
                    # opponent move if mover toggles each ply
                    opp_move = (mover_is_white[j] != mover_white)
                    if opp_move and mistakes[j] == 1:
                        hit = 1
                    vmax = max(vmax, abs(cp_before[j] - base))
                opp_mistake_soon[i] = float(hit)
                volatility[i] = clamp01(vmax / 300.0)

            # Build labeled rows
            for i, rec in enumerate(g):
                board = chess.Board(rec["fen"])
                legal = board.legal_moves.count()
                comp = clamp01(legal / 40.0)

                # value target from stockfish cp_before (already mover perspective)
                v = tanh_value(cp_before[i])

                # pressure: does opponent blunder soon (within horizon)
                p = opp_mistake_soon[i]

                # complexity: proxy
                x = comp

                # volatility: local eval swing
                vol = volatility[i]

                # weight: emphasize positions that lead to mistakes
                weight = 1.0 + 2.0 * p

                pol_obj = None
                if cfg.multipv > 0:
                    mvs = eval_board_multipv(sf, board, cfg)
                    idxs: list[int] = []
                    cps: list[int] = []
                    for mv, cp in mvs:
                        if mv not in board.legal_moves:
                            continue
                        idx = move_to_index(board, mv)
                        if idx < 0 or idx >= MOVE_SPACE:
                            continue
                        idxs.append(int(idx))
                        cps.append(int(cp))
                    probs = softmax_cps(cps, cfg.policy_temp)
                    if idxs and probs:
                        pol_obj = {"idx": idxs, "p": probs}

                labeled_rows.append({
                    "run_id": run_id,
                    "game_id": gid,
                    "ply": int(rec["ply"]),
                    "fen": rec["fen"],
                    "planes": rec.get("planes") or encode_planes(board),
                    "policy": pol_obj,
                    "targets": {
                        "value": v,
                        "pressure": p,
                        "volatility": vol,
                        "complexity": x,
                    },
                    "weight": weight,
                    "meta": {
                        "cp_before": cp_before[i],
                        "mistake_mover": mistakes[i],
                        "legal_moves": legal,
                        "result": rec.get("result", ""),
                    }
                })

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in labeled_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    finally:
        sf.quit()

    append_event(root, {"ts_ms": now_ms(), "type": "rl_label_done", "run_id": run_id, "out": str(out), "rows": len(labeled_rows)})
    print(f"Done. Labeled: {out} ({len(labeled_rows)} rows)")


if __name__ == "__main__":
    main()
