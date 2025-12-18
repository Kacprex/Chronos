import random
import chess

# VERY SMALL, SAFE OPENING SET
OPENINGS = [
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
]

def play_random_opening(board: chess.Board, max_plies=6):
    opening = random.choice(OPENINGS)
    moves_played = 0

    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)
            moves_played += 1
        if moves_played >= max_plies:
            break

    return moves_played
