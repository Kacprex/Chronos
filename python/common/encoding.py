from __future__ import annotations
import numpy as np
from .fen import parse_fen, PIECE_PLANES

PLANES = 18
H = 8
W = 8
INPUT_DIM = PLANES * H * W

def encode_fen_18x8x8(fen_str: str) -> np.ndarray:
    f = parse_fen(fen_str)
    x = np.zeros((PLANES, H, W), dtype=np.float32)

    for r in range(8):
        for c in range(8):
            p = f.board[r][c]
            pl = PIECE_PLANES.get(p, None)
            if pl is not None:
                x[pl, r, c] = 1.0

    # side to move
    if f.white_to_move:
        x[12, :, :] = 1.0

    # castling
    if f.castle_wk:
        x[13, :, :] = 1.0
    if f.castle_wq:
        x[14, :, :] = 1.0
    if f.castle_bk:
        x[15, :, :] = 1.0
    if f.castle_bq:
        x[16, :, :] = 1.0

    # halfmove clock normalized
    hm = min(max(f.halfmove_clock, 0), 100) / 100.0
    x[17, :, :] = hm

    return x.reshape(-1)  # 1152
