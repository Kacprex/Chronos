"""Backwards-compatible ChessNet import.

The Chronos network implementation is in `src.nn.network`.
Some older patches import `ChessNet` from `src.model.chess_model`.
This shim keeps that import path working.
"""

from __future__ import annotations

from src.nn.network import ChessNet  # re-export

__all__ = ["ChessNet"]
