from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Iterable

import chess
import chess.pgn
from tqdm import tqdm


def default_root() -> Path:
    return Path(os.environ.get("CHRONOS_DATA_ROOT", r"E:/chronos"))


@dataclass
class ExtractConfig:
    pgn_path: Path
    out_jsonl: Path
    min_moves: int = 40
    max_games: int = 0  # 0 = all
    shard_start_move: int = 12
    shard_end_move: int = 60


PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_planes(board: chess.Board) -> list[float]:
    '''
    Must match engine encoder in shape: 25*64, plane-major.

    Plane order:
      0..5  White P,N,B,R,Q,K
      6..11 Black P,N,B,R,Q,K
      12    side to move plane (1.0 if white to move else 0.0)
      13..16 castling KQkq constant planes
      17..24 ep-file planes (8)
    '''
    planes = [0.0] * (25 * 64)

    def set_plane_sq(p: int, sq: int, v: float = 1.0) -> None:
        planes[p * 64 + sq] = v

    # pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        base = 0 if piece.color == chess.WHITE else 6
        pl = base + PIECE_TO_PLANE[piece.piece_type]
        set_plane_sq(pl, sq, 1.0)

    # side to move
    stm = 1.0 if board.turn == chess.WHITE else 0.0
    for sq in chess.SQUARES:
        set_plane_sq(12, sq, stm)

    # castling planes
    def fill_plane(p: int, val: float) -> None:
        for sq in chess.SQUARES:
            set_plane_sq(p, sq, val)

    fill_plane(13, 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    fill_plane(14, 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    fill_plane(15, 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    fill_plane(16, 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    # ep file planes
    ep = board.ep_square
    for f in range(8):
        val = 0.0
        if ep is not None and chess.square_file(ep) == f:
            val = 1.0
        fill_plane(17 + f, val)

    return planes


def result_value(result: str, stm_is_white: bool) -> float:
    if result == "1-0":
        return 1.0 if stm_is_white else -1.0
    if result == "0-1":
        return -1.0 if stm_is_white else 1.0
    return 0.0


def iter_games(pgn_file: Path) -> Iterable[chess.pgn.Game]:
    with pgn_file.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Input PGN file")
    ap.add_argument("--out", default="", help="Output JSONL. Default: E:/chronos/shards/sl/<name>/shards.jsonl")
    ap.add_argument("--name", default="", help="Dataset name folder under shards/sl. Default: pgn stem + timestamp")
    ap.add_argument("--min-moves", type=int, default=40)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--start-move", type=int, default=12)
    ap.add_argument("--end-move", type=int, default=60)
    args = ap.parse_args()

    root = default_root()
    pgn_path = Path(args.pgn)

    name = args.name or f"{pgn_path.stem}_{int(time())}"
    out_path = Path(args.out) if args.out else (root / "shards" / "sl" / name / "shards.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ExtractConfig(
        pgn_path=pgn_path,
        out_jsonl=out_path,
        min_moves=args.min_moves,
        max_games=args.max_games,
        shard_start_move=args.start_move,
        shard_end_move=args.end_move,
    )

    n_games = 0
    n_rows = 0

    with cfg.out_jsonl.open("w", encoding="utf-8") as out:
        for game in tqdm(iter_games(cfg.pgn_path), desc="PGN games"):
            n_games += 1
            if cfg.max_games and n_games > cfg.max_games:
                break

            moves = list(game.mainline_moves())
            if len(moves) < cfg.min_moves:
                continue

            result = game.headers.get("Result", "1/2-1/2")

            board = game.board()

            for move_index, mv in enumerate(moves):
                move_no = move_index + 1
                if move_no < cfg.shard_start_move or move_no > cfg.shard_end_move:
                    board.push(mv)
                    continue

                legal_count = board.legal_moves.count()
                comp = min(1.0, legal_count / 40.0)
                val = result_value(result, board.turn == chess.WHITE)

                row = {
                    "game_id": n_games,
                    "move_no": move_no,
                    "fen": board.fen(),
                    "planes": encode_planes(board),
                    "meta": {
                        "result": result,
                        "legal_moves": legal_count,
                    },
                    "targets": {
                        "value": val,
                        "pressure": comp,     # bootstrap proxy
                        "volatility": 0.0,    # placeholder
                        "complexity": comp,
                    },
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

                board.push(mv)

    print(f"Wrote {n_rows} shard rows to: {cfg.out_jsonl}")
    print(f"Games scanned: {n_games}")


if __name__ == "__main__":
    main()
