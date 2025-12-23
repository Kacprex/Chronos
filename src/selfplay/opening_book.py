"""Opening selection helper.

Used by self-play and promotion to avoid deterministic collapse.

Optional external book: set CHRONOS_OPENING_BOOK_PATH to a text file.
Supported line formats (whitespace separated):
- UCI moves:            e2e4 e7e5 g1f3 b8c6
- with prefix:          uci: e2e4 e7e5 ...
- FEN start position:   fen: <full FEN string>
Blank lines and lines starting with # are ignored.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess


# Optional external opening book path.
# If empty/missing, we fall back to the builtin opening list.
OPENING_BOOK_PATH = os.environ.get("CHRONOS_OPENING_BOOK_PATH", "").strip()


@dataclass(frozen=True)
class Opening:
    name: str
    uci_moves: Tuple[str, ...] = ()
    fen: str = ""  # if set, overrides start position
    source: str = "builtin"


# A deliberately broad set of short openings (2-6 plies) to push variety.
_BUILTIN: List[Opening] = [
    Opening("King's Pawn", ("e2e4", "e7e5")),
    Opening("Sicilian", ("e2e4", "c7c5")),
    Opening("French", ("e2e4", "e7e6")),
    Opening("Caro-Kann", ("e2e4", "c7c6")),
    Opening("Pirc", ("e2e4", "d7d6", "d2d4", "g8f6")),
    Opening("Modern", ("e2e4", "g7g6")),
    Opening("Scandinavian", ("e2e4", "d7d5")),
    Opening("Alekhine", ("e2e4", "g8f6")),
    Opening("King's Indian Attack", ("g1f3", "d7d5", "g2g3", "g8f6")),
    Opening("Italian", ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4")),
    Opening("Spanish", ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5")),
    Opening("Scotch", ("e2e4", "e7e5", "g1f3", "b8c6", "d2d4")),
    Opening("Vienna", ("e2e4", "e7e5", "b1c3")),
    Opening("King's Gambit", ("e2e4", "e7e5", "f2f4")),
    Opening("Queen's Pawn", ("d2d4", "d7d5")),
    Opening("Queen's Gambit", ("d2d4", "d7d5", "c2c4")),
    Opening("Slav", ("d2d4", "d7d5", "c2c4", "c7c6")),
    Opening("QGD", ("d2d4", "d7d5", "c2c4", "e7e6")),
    Opening("Nimzo", ("d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4")),
    Opening("Queen's Indian", ("d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6")),
    Opening("Catalan", ("d2d4", "g8f6", "c2c4", "e7e6", "g2g3")),
    Opening("English", ("c2c4", "e7e5")),
    Opening("English Sym", ("c2c4", "c7c5")),
    Opening("Reti", ("g1f3", "d7d5", "c2c4")),
    Opening("Bird", ("f2f4", "d7d5")),
    Opening("Dutch", ("d2d4", "f7f5")),
    Opening("Benoni", ("d2d4", "g8f6", "c2c4", "c7c5", "d4d5")),
    Opening("Benko", ("d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5")),
    Opening("London", ("d2d4", "d7d5", "g1f3", "g8f6", "c1f4")),
    Opening("Trompowsky", ("d2d4", "g8f6", "c1g5")),
    Opening("Jobava", ("d2d4", "d7d5", "b1c3")),
    Opening("Polish", ("b2b4",)),
    Opening("Sokolsky", ("b2b4", "e7e5")),
    Opening("Grob", ("g2g4",)),
]


def _parse_external_line(line: str) -> Optional[Opening]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    if s.lower().startswith("fen:"):
        fen = s[4:].strip()
        if not fen:
            return None
        # Name = first 24 chars of FEN for readability
        name = f"FEN:{fen.split(' ')[0]}"
        return Opening(name=name, fen=fen, source="external")

    if s.lower().startswith("uci:"):
        s = s[4:].strip()

    parts = s.split()
    if not parts:
        return None

    # Treat as UCI move list
    return Opening(name=f"EXT:{parts[0]}", uci_moves=tuple(parts), source="external")


def load_openings(book_path: str) -> List[Opening]:
    if not book_path:
        return list(_BUILTIN)

    try:
        with open(book_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        # Fail open: fall back to builtin
        return list(_BUILTIN)

    ext: List[Opening] = []
    for ln in lines:
        op = _parse_external_line(ln)
        if op is not None:
            ext.append(op)

    # If external is empty, still keep builtin
    return ext if ext else list(_BUILTIN)


def apply_opening(
    board: chess.Board,
    opening: Opening,
    max_plies: int,
    random_cut: bool = True,
) -> Tuple[str, List[str]]:
    """Apply an opening to a board.

    Returns (opening_name, applied_uci_moves).
    """
    applied: List[str] = []

    if opening.fen:
        try:
            board.set_fen(opening.fen)
        except Exception:
            # If FEN fails, keep starting position
            pass

    moves = list(opening.uci_moves)
    if not moves or max_plies <= 0:
        return opening.name, applied

    # Cut length: 2..max_plies, biased towards shorter to keep variety.
    if random_cut:
        k = min(len(moves), max_plies)
        if k >= 2:
            k = random.randint(2, k)
        moves = moves[:k]
    else:
        moves = moves[:max_plies]

    for uci in moves:
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            break
        if mv not in board.legal_moves:
            break
        board.push(mv)
        applied.append(uci)

    return opening.name, applied


def sample_opening(
    board: "chess.Board",
    max_plies: int = 8,
    book_path: str | None = None,
    rng: random.Random | None = None,
):
    """Apply a random opening line from the book to an existing board.

    - board is modified in-place.
    - Returns a small info dict (or None if no opening was applied).

    This is used by promotion games to randomize starting positions so
    latest vs best doesn't repeatedly hit the exact same openings.
    """
    if board is None:
        return None

    if rng is None:
        rng = random

    if book_path is None:
        # Fall back to default path used by load_openings()
        book_path = OPENING_BOOK_PATH

    openings = load_openings(book_path)
    if not openings:
        return None

    opening = rng.choice(openings)
    name, applied = apply_opening(board, opening, max_plies=max_plies, random_cut=True)
    plies_applied = len(applied)
    if plies_applied <= 0:
        return None

    return {
        "name": name,
        "plies_applied": plies_applied,
        "uci_moves": applied,
        "source": getattr(opening, "source", "builtin"),
    }


def new_game_board(
    max_plies: int,
    book_path: Optional[str] = None,
    use_openings: bool = True,
) -> Tuple[chess.Board, str, List[str]]:
    """Create a fresh board and optionally apply a random opening."""
    board = chess.Board()
    if not use_openings or max_plies <= 0:
        return board, "(none)", []

    if book_path is None:
        book_path = os.environ.get("CHRONOS_OPENING_BOOK_PATH", "").strip()

    openings = load_openings(book_path)
    opening = random.choice(openings)
    name, applied = apply_opening(board, opening, max_plies=max_plies, random_cut=True)
    return board, name, applied


# Backward-compatible alias
def play_random_opening(board, *, book_path: str = 'data/openings.pgn', max_plies: int = 6):
    openings = load_openings(book_path)
    if not openings:
        return "(none)", []
    opening = random.choice(openings)
    return apply_opening(board, opening, max_plies=max_plies, random_cut=True)
