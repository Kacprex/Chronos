from __future__ import annotations

import chess

MOVE_TYPES = 73
MOVE_SPACE = 64 * MOVE_TYPES

# Directions (absolute on board coordinates)
DIRS = [
    (0, 1),   # N
    (0, -1),  # S
    (1, 0),   # E
    (-1, 0),  # W
    (1, 1),   # NE
    (-1, 1),  # NW
    (1, -1),  # SE
    (-1, -1), # SW
]

KN = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

def _promo_group(promo: int | None) -> int:
    if promo == chess.KNIGHT:
        return 0
    if promo == chess.BISHOP:
        return 1
    if promo == chess.ROOK:
        return 2
    return -1  # queen or none

def move_to_index(board: chess.Board, move: chess.Move) -> int:
    '''
    AlphaZero-style 8x8x73 move index.
    Returns -1 if not representable.
    '''
    f = move.from_square
    t = move.to_square
    ff, fr = chess.square_file(f), chess.square_rank(f)
    tf, tr = chess.square_file(t), chess.square_rank(t)
    df, dr = tf - ff, tr - fr

    from_idx = int(f)
    if from_idx < 0 or from_idx >= 64:
        return -1

    # Underpromotions (N,B,R only), relative to side-to-move
    piece = board.piece_at(f)
    if piece and piece.piece_type == chess.PAWN:
        pg = _promo_group(move.promotion)
        if pg >= 0:
            fwd_dr = 1 if board.turn == chess.WHITE else -1
            if df == 0 and dr == fwd_dr:
                dir_idx = 0
            else:
                if board.turn == chess.WHITE:
                    capL = (df == -1 and dr == 1)
                    capR = (df == 1 and dr == 1)
                else:
                    capL = (df == 1 and dr == -1)
                    capR = (df == -1 and dr == -1)
                if capL:
                    dir_idx = 1
                elif capR:
                    dir_idx = 2
                else:
                    return -1
            typ = 64 + pg * 3 + dir_idx
            return from_idx * MOVE_TYPES + typ

    # Knights
    for i, (kx, ky) in enumerate(KN):
        if df == kx and dr == ky:
            return from_idx * MOVE_TYPES + (56 + i)

    # Queen-like moves
    adf, adr = abs(df), abs(dr)
    dist = max(adf, adr)
    if dist < 1 or dist > 7:
        return -1
    sdf = 0 if df == 0 else df // adf
    sdr = 0 if dr == 0 else dr // adr
    for d, (dx, dy) in enumerate(DIRS):
        if sdf == dx and sdr == dy:
            typ = d * 7 + (dist - 1)
            return from_idx * MOVE_TYPES + typ

    return -1
