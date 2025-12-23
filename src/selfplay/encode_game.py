from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import chess
import chess.pgn


@dataclass
class GameRecord:
    """
    Lightweight container for a single game:
    - list of moves (UCI)
    - result string: '1-0', '0-1', '1/2-1/2', '*'
    - metadata for PGN tags (including optional MCTS sims + SF depth)
    """
    moves_uci: List[str] = field(default_factory=list)
    result: str = "1/2-1/2"
    white_name: str = "SelfPlay-White"
    black_name: str = "SelfPlay-Black"
    event: str = "SelfPlay"
    site: str = "Local"
    date: Optional[str] = None  # if None, use today's date (UTC)
    mcts_sims: Optional[int] = None
    sf_depth: Optional[int] = None

    def to_chess_game(self, extra_headers: Optional[Dict[str, Any]] = None) -> chess.pgn.Game:
        """Convert this record into a python-chess Game object, with headers."""
        game = chess.pgn.Game()

        game.headers["Event"] = self.event
        game.headers["Site"] = self.site
        game.headers["Date"] = self.date or datetime.utcnow().strftime("%Y.%m.%d")
        game.headers["Round"] = "-"
        game.headers["White"] = self.white_name
        game.headers["Black"] = self.black_name
        game.headers["Result"] = self.result

        # Standardized extra tags
        if self.mcts_sims is not None:
            game.headers["MCTS"] = str(self.mcts_sims)
        if self.sf_depth is not None:
            game.headers["SFDepth"] = str(self.sf_depth)

        # User-provided extra headers (e.g., temperature, exploration, seed, etc.)
        if extra_headers:
            for k, v in extra_headers.items():
                if v is None:
                    continue
                game.headers[str(k)] = str(v)

        board = chess.Board()
        node = game

        for uci in self.moves_uci:
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in board.legal_moves:
                # If illegal due to some bug, stop encoding further moves
                break
            board.push(move)
            node = node.add_variation(move)

        return game

    def to_pgn_string(self, extra_headers: Optional[Dict[str, Any]] = None) -> str:
        """Return the PGN representation as a string."""
        game = self.to_chess_game(extra_headers=extra_headers)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return game.accept(exporter)

    # Backward-compat alias (some of your older code called .to_pgn())
    def to_pgn(self, extra_headers: Optional[Dict[str, Any]] = None) -> str:
        return self.to_pgn_string(extra_headers=extra_headers)

    def save_pgn(self, path: str, extra_headers: Optional[Dict[str, Any]] = None) -> None:
        """
        Append this game as PGN to `path`.
        Fixes WinError 3 when path has no directory component (e.g. "ai_vs_ai_games.pgn").
        """
        # Only create directories if a directory part exists.
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        pgn_text = self.to_pgn_string(extra_headers=extra_headers)

        with open(path, "a", encoding="utf-8") as f:
            f.write(pgn_text)
            f.write("\n\n")


def build_game_record_from_moves(
    moves: List[chess.Move],
    result: str,
    white_name: str = "SelfPlay-White",
    black_name: str = "SelfPlay-Black",
    event: str = "SelfPlay",
    site: str = "Local",
    date: Optional[str] = None,
    mcts_sims: Optional[int] = None,
    sf_depth: Optional[int] = None,
) -> GameRecord:
    """
    Helper: build a GameRecord from a list of chess.Move objects and a result string.
    """
    moves_uci = [mv.uci() for mv in moves]
    return GameRecord(
        moves_uci=moves_uci,
        result=result,
        white_name=white_name,
        black_name=black_name,
        event=event,
        site=site,
        date=date,
        mcts_sims=mcts_sims,
        sf_depth=sf_depth,
    )
