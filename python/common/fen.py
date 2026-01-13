from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Fen:
    # board ranks 8->1 as rows 0..7, files a->h as cols 0..7
    board: list[list[str]]
    white_to_move: bool
    castle_wk: bool
    castle_wq: bool
    castle_bk: bool
    castle_bq: bool
    halfmove_clock: int
    fullmove_number: int

PIECE_PLANES = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

def parse_fen(fen_str: str) -> Fen:
    parts = fen_str.strip().split()
    if len(parts) < 4:
        raise ValueError("Invalid FEN: missing fields")

    placement, stm, castling, ep = parts[:4]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    board = [["." for _ in range(8)] for _ in range(8)]
    r = 0
    c = 0
    for ch in placement:
        if ch == "/":
            r += 1
            c = 0
            continue
        if ch.isdigit():
            n = int(ch)
            for _ in range(n):
                board[r][c] = "."
                c += 1
        else:
            board[r][c] = ch
            c += 1

    white_to_move = (stm == "w")
    castle_wk = "K" in castling
    castle_wq = "Q" in castling
    castle_bk = "k" in castling
    castle_bq = "q" in castling

    return Fen(
        board=board,
        white_to_move=white_to_move,
        castle_wk=castle_wk,
        castle_wq=castle_wq,
        castle_bk=castle_bk,
        castle_bq=castle_bq,
        halfmove_clock=halfmove,
        fullmove_number=fullmove,
    )
