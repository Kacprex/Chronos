from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import chess

from nn.move_index import move_to_index, MOVE_SPACE


@dataclass
class EdgeStats:
    n: int = 0
    w: float = 0.0  # total value from current player's POV (at this node)
    p: float = 0.0  # prior

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


@dataclass
class Node:
    board_fen: str
    to_play_white: bool
    edges: Dict[chess.Move, EdgeStats] = field(default_factory=dict)
    children: Dict[chess.Move, "Node"] = field(default_factory=dict)
    expanded: bool = False
    terminal_value: Optional[float] = None  # from current player's POV

    def is_terminal(self) -> bool:
        return self.terminal_value is not None


def outcome_value(board: chess.Board) -> Optional[float]:
    if not board.is_game_over(claim_draw=True):
        return None
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    if res == "0-1":
        return 1.0 if board.turn == chess.BLACK else -1.0
    return 0.0


def dirichlet_noise(size: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet([alpha] * size)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if s <= 0:
        return np.ones_like(x) / len(x)
    return (e / s).astype(np.float32)


def expand_node(node: Node, board: chess.Board, priors_logits: np.ndarray, add_noise: bool, rng: np.random.Generator, noise_alpha: float, noise_frac: float) -> None:
    legal = list(board.legal_moves)
    if not legal:
        node.terminal_value = outcome_value(board) or 0.0
        return

    moves: List[chess.Move] = []
    logits: List[float] = []
    for mv in legal:
        idx = move_to_index(board, mv)
        if idx < 0 or idx >= MOVE_SPACE:
            continue
        moves.append(mv)
        logits.append(float(priors_logits[idx]))

    if not moves:
        p = 1.0 / len(legal)
        node.edges = {mv: EdgeStats(n=0, w=0.0, p=p) for mv in legal}
        node.expanded = True
        return

    probs = softmax(np.array(logits, dtype=np.float32))
    if add_noise:
        noise = dirichlet_noise(len(moves), noise_alpha, rng)
        probs = (1.0 - noise_frac) * probs + noise_frac * noise

    node.edges = {mv: EdgeStats(n=0, w=0.0, p=float(p)) for mv, p in zip(moves, probs.tolist())}
    node.expanded = True


def select_child(node: Node, c_puct: float) -> chess.Move:
    sum_n = sum(es.n for es in node.edges.values())
    sqrt_sum = np.sqrt(max(1.0, float(sum_n)))

    best_mv: Optional[chess.Move] = None
    best_score = -1e9

    for mv, es in node.edges.items():
        u = c_puct * es.p * sqrt_sum / (1.0 + es.n)
        score = es.q + u
        if score > best_score:
            best_score = score
            best_mv = mv

    assert best_mv is not None
    return best_mv


def backup(path: List[Tuple[Node, chess.Move]], leaf_value: float) -> None:
    # leaf_value is from leaf node current player's POV
    v = leaf_value
    for node, mv in reversed(path):
        es = node.edges[mv]
        es.n += 1
        es.w += v
        v = -v  # flip POV each ply


def policy_from_visits(root: Node) -> Tuple[List[int], List[float]]:
    base_board = chess.Board(root.board_fen)
    idxs: List[int] = []
    ps: List[float] = []
    total = 0
    for mv, es in root.edges.items():
        if es.n <= 0:
            continue
        idx = move_to_index(base_board, mv)
        if idx < 0 or idx >= MOVE_SPACE:
            continue
        idxs.append(idx)
        ps.append(float(es.n))
        total += es.n
    if total <= 0:
        return [], []
    ps = [p / total for p in ps]
    return idxs, ps
