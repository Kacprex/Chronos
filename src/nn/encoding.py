import chess
import torch
import numpy as np


MOVE_SPACE = 4672  # Full AlphaZero-style move indexing


def encode_board(board: chess.Board):
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # 12 planes for pieces
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        piece_plane = piece_map[piece.piece_type] + color_offset
        row, col = divmod(square, 8)
        planes[piece_plane, 7 - row, col] = 1.0

    # Side to move
    planes[12].fill(1.0 if board.turn == chess.WHITE else 0.0)

    # Castling rights
    planes[13].fill(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[14].fill(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[15].fill(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[16].fill(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    # Halfmove clock
    planes[17].fill(board.halfmove_clock / 100.0)

    return planes


def move_to_index(move: chess.Move):
    """
    AlphaZero move indexing:
    73 planes * 8 * 8 = 4672 moves
    """

    start_sq = move.from_square
    end_sq = move.to_square
    start_rank, start_file = divmod(start_sq, 8)
    end_rank, end_file = divmod(end_sq, 8)

    direction = end_file - start_file
    distance = abs(end_rank - start_rank) + abs(end_file - start_file)

    # Promotion handling:
    if move.promotion:
        promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
        return 4096 + start_sq * 4 + promo_map[move.promotion]

    # Simplified indexing for now
    idx = start_sq * 64 + end_sq
    return idx if idx < MOVE_SPACE else None
def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert a policy index back into a legal chess.Move
    by scanning current legal moves.
    """

    for move in board.legal_moves:
        if move_to_index(move) == index:
            return move

    raise ValueError(f"No legal move found for index {index}")