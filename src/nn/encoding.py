import chess
import numpy as np

# AlphaZero-style move indexing: 73 move-types * 64 from-squares = 4672
MOVE_SPACE = 4672

# Move-type planes:
# 0..55  : 8 directions * 7 distances (queen-like moves)
# 56..63 : 8 knight moves
# 64..72 : 9 underpromotions (3 directions: forward, capture-left, capture-right) * 3 pieces (N,B,R)
#
# Note: Queen promotions are encoded as the corresponding queen-like move (distance=1).


# Direction order for queen-like moves (dx, dy) in file/rank coordinates.
# file: a->h is +1, rank: 1->8 is +1
_Q_DIRS = [
    (0, 1),   # N
    (0, -1),  # S
    (1, 0),   # E
    (-1, 0),  # W
    (1, 1),   # NE
    (-1, 1),  # NW
    (1, -1),  # SE
    (-1, -1), # SW
]
_Q_DIR_TO_IDX = {d: i for i, d in enumerate(_Q_DIRS)}

# Knight move order (dx, dy)
_K_DIRS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]
_K_DIR_TO_IDX = {d: i for i, d in enumerate(_K_DIRS)}

_UNDERPROMO_PIECES = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board into (18, 8, 8) float32 planes.

    Planes:
      0..5   white pieces (P,N,B,R,Q,K)
      6..11  black pieces (P,N,B,R,Q,K)
      12     side to move (all 1 if white to move else 0)
      13..16 castling rights (WK, WQ, BK, BQ)
      17     halfmove clock / 100, with an en-passant marker (+1 at ep square if present)
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for sq, piece in board.piece_map().items():
        base = piece_map[piece.piece_type]
        plane = base if piece.color == chess.WHITE else base + 6
        row, col = divmod(sq, 8)
        planes[plane, 7 - row, col] = 1.0

    planes[12].fill(1.0 if board.turn == chess.WHITE else 0.0)

    planes[13].fill(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[14].fill(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[15].fill(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[16].fill(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    # Halfmove clock baseline
    baseline = board.halfmove_clock / 100.0
    planes[17].fill(baseline)

    # En-passant marker (keeps 18 planes; adds a distinct +1 spike at ep square)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        planes[17, 7 - row, col] += 1.0

    return planes


def _move_type_plane(move: chess.Move) -> int | None:
    """Return move-type plane in [0,72] for AlphaZero-style encoding."""
    from_sq = move.from_square
    to_sq = move.to_square

    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    tx, ty = chess.square_file(to_sq), chess.square_rank(to_sq)
    dx, dy = tx - fx, ty - fy

    adx, ady = abs(dx), abs(dy)

    # Underpromotions (N/B/R) have dedicated planes.
    if move.promotion in _UNDERPROMO_PIECES:
        # dy is +1 for white promotions, -1 for black promotions
        if dy not in (1, -1):
            return None
        # Direction relative to pawn forward:
        # forward: dx==0
        # capture-left: dx == -dy
        # capture-right: dx == dy
        if dx == 0:
            dir_idx = 0
        elif dx == -dy:
            dir_idx = 1
        elif dx == dy:
            dir_idx = 2
        else:
            return None

        piece_idx = _UNDERPROMO_PIECES[move.promotion]
        return 64 + piece_idx * 3 + dir_idx  # 64..72

    # Knight moves
    if (adx, ady) in ((1, 2), (2, 1)):
        k = _K_DIR_TO_IDX.get((dx, dy))
        if k is None:
            return None
        return 56 + k  # 56..63

    # Queen-like moves: straight or diagonal
    if dx == 0 and dy != 0:
        step = (0, 1 if dy > 0 else -1)
        dist = ady
    elif dy == 0 and dx != 0:
        step = (1 if dx > 0 else -1, 0)
        dist = adx
    elif adx == ady and adx != 0:
        step = (1 if dx > 0 else -1, 1 if dy > 0 else -1)
        dist = adx
    else:
        return None

    dir_idx = _Q_DIR_TO_IDX.get(step)
    if dir_idx is None or dist < 1 or dist > 7:
        return None
    return dir_idx * 7 + (dist - 1)  # 0..55


def move_to_index(move: chess.Move) -> int | None:
    """Map a chess.Move to an integer index in [0, MOVE_SPACE).

    Encoding: index = plane * 64 + from_square
    where plane is in [0,72] and from_square in [0,63].
    """
    plane = _move_type_plane(move)
    if plane is None:
        return None
    return plane * 64 + move.from_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a policy index back into a legal chess.Move by scanning legal moves.

    We filter by from_square to make this faster than scanning all moves.
    """
    if index < 0 or index >= MOVE_SPACE:
        raise ValueError(f"Index out of bounds: {index}")

    plane = index // 64
    from_sq = index % 64

    for move in board.legal_moves:
        if move.from_square != from_sq:
            continue
        if _move_type_plane(move) == plane:
            return move

    raise ValueError(f"No legal move found for index {index}")
