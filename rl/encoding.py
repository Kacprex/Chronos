from __future__ import annotations

import chess

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
    25 planes, plane-major, 64 squares each, matching engine Phase 6 encoder.

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

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        base = 0 if piece.color == chess.WHITE else 6
        pl = base + PIECE_TO_PLANE[piece.piece_type]
        set_plane_sq(pl, sq, 1.0)

    stm = 1.0 if board.turn == chess.WHITE else 0.0
    for sq in chess.SQUARES:
        set_plane_sq(12, sq, stm)

    def fill_plane(p: int, val: float) -> None:
        for sq in chess.SQUARES:
            set_plane_sq(p, sq, val)

    fill_plane(13, 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    fill_plane(14, 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    fill_plane(15, 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    fill_plane(16, 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    ep = board.ep_square
    for f in range(8):
        val = 0.0
        if ep is not None and chess.square_file(ep) == f:
            val = 1.0
        fill_plane(17 + f, val)

    return planes
