"""Compatibility package.

Some older code paths (especially in promotion/evaluation patches) referenced
`src.model.chess_model.ChessNet`. The current Chronos code keeps the network in
`src.nn.network`.

This package provides a stable import path so old code doesn't crash.
"""

from .chess_model import ChessNet  # re-export
