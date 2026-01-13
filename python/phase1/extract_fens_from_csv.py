from __future__ import annotations

import argparse
import io
import os
import random
from pathlib import Path

import pandas as pd
import chess.pgn
from tqdm import tqdm


def iter_fens_from_pgn(pgn_text: str, sample_every: int, start_ply: int, max_fens_per_game: int):
    """
    Yield sampled FENs from a single PGN.
    - sample_every: take every N plies
    - start_ply: ignore early opening plies (reduces duplicates)
    - max_fens_per_game: cap to avoid very long games dominating
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return

    board = game.board()
    ply = 0
    taken = 0

    for move in game.mainline_moves():
        board.push(move)
        ply += 1

        if ply < start_ply:
            continue

        if (ply - start_ply) % sample_every == 0:
            yield board.fen()  # full FEN incl. side-to-move/castling/halfmove/fullmove
            taken += 1
            if taken >= max_fens_per_game:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV containing a 'pgn' column")
    ap.add_argument("--out_fens", required=True, help="Output raw_fens.txt path")
    ap.add_argument("--chunksize", type=int, default=5000)
    ap.add_argument("--sample_every", type=int, default=4, help="sample every N plies")
    ap.add_argument("--start_ply", type=int, default=8, help="skip early plies to reduce duplicates")
    ap.add_argument("--max_fens_per_game", type=int, default=40)
    ap.add_argument("--max_games", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max_fens", type=int, default=0, help="0 = no limit")
    ap.add_argument("--min_elo", type=int, default=0, help="filter by max(WhiteElo, BlackElo) >= min_elo")
    ap.add_argument("--pgn_col", type=str, default="pgn")
    ap.add_argument("--dedup", action="store_true", help="deduplicate FENs (memory heavy if huge)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)

    csv_path = Path(args.csv)
    out_path = Path(args.out_fens)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # For very large runs, dedup=False is recommended (or do external dedup later).
    seen = set() if args.dedup else None

    total_games = 0
    total_fens = 0
    written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for chunk in pd.read_csv(csv_path, chunksize=args.chunksize):
            # Basic sanity: ensure column exists
            if args.pgn_col not in chunk.columns:
                raise ValueError(f"CSV missing column '{args.pgn_col}'. Available: {list(chunk.columns)}")

            for _, row in chunk.iterrows():
                if args.max_games and total_games >= args.max_games:
                    break
                if args.max_fens and written >= args.max_fens:
                    break

                total_games += 1

                # Optional Elo filter
                if args.min_elo:
                    we = int(row.get("WhiteElo", 0) or 0)
                    be = int(row.get("BlackElo", 0) or 0)
                    if max(we, be) < args.min_elo:
                        continue

                pgn_text = row.get(args.pgn_col, None)
                if not isinstance(pgn_text, str) or not pgn_text.strip():
                    continue

                try:
                    for fen in iter_fens_from_pgn(
                        pgn_text,
                        sample_every=args.sample_every,
                        start_ply=args.start_ply,
                        max_fens_per_game=args.max_fens_per_game,
                    ):
                        total_fens += 1
                        if args.max_fens and written >= args.max_fens:
                            break

                        if seen is not None:
                            if fen in seen:
                                continue
                            seen.add(fen)

                        out.write(fen + "\n")
                        written += 1

                except Exception:
                    # Skip malformed PGN rows
                    continue

            if args.max_games and total_games >= args.max_games:
                break
            if args.max_fens and written >= args.max_fens:
                break

    print(f"Done. Games processed: {total_games}, FENs written: {written}, FENs seen (raw): {total_fens}")


if __name__ == "__main__":
    main()
