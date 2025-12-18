import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import torch
import chess
import numpy as np
from src.nn.encoding import encode_board, move_to_index, MOVE_SPACE


# ===================== NODE =====================

@dataclass
class Node:
    board: chess.Board
    parent: Optional["Node"] = None
    move: Optional[chess.Move] = None

    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0

    children: Dict[chess.Move, "Node"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def expand(self, policy_logits: torch.Tensor):
        if self.expanded or self.is_terminal():
            return

        policy = torch.softmax(policy_logits, dim=0)

        for move in self.board.legal_moves:
            idx = move_to_index(move)
            if idx is None:
                continue

            child_board = self.board.copy(stack=False)
            child_board.push(move)

            self.children[move] = Node(
                board=child_board,
                parent=self,
                move=move,
                prior=float(policy[idx]),
            )

        self.expanded = True


# ===================== MCTS =====================

class MCTS:
    def __init__(
        self,
        model,
        device,
        simulations: int = 400,
        cpuct: float = 1.5,
        add_dirichlet_noise: bool = True,
    ):
        self.model = model
        self.device = device
        self.simulations = simulations

        # --- UCT ---
        self.cpuct = cpuct

        # --- Dirichlet noise (root only) ---
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25

        # --- Temperature schedule ---
        self.temp_initial = 1.25
        self.temp_moves = 20   # first 20 plies exploratory

        # --- Random opening book ---
        self.opening_random_plies = (4, 6)  # inclusive range

        # --- Internal ---
        self.children = {}



    def run(
        self,
        board: chess.Board,
        move_number: int,
        add_noise: bool = True,
    ):
        """
        Returns:
            moves: List[chess.Move]
            probs: np.ndarray (len == len(moves))
        """

        # ================= RANDOM OPENING =================
        if add_noise and move_number < random.randint(*self.opening_random_plies):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None, None

            move = random.choice(legal_moves)
            return [move], np.array([1.0], dtype=np.float32)

        # ================= ROOT =================
        root = Node(board.copy(stack=False))

        # Expand root once
        self._expand(root)

        # Dirichlet noise (root only)
        if add_noise:
            self._add_dirichlet_noise(root)

        # ================= SIMULATIONS =================
        for _ in range(self.simulations):
            node = root

            # Selection
            while node.expanded and not node.is_terminal():
                node = self._select(node)

            # Evaluation / Expansion
            if node.is_terminal():
                value = self._terminal_value(node.board)
                self._backpropagate(node, value)
            else:
                value = self._expand(node)
                self._backpropagate(node, value)

        # ================= POLICY FROM VISITS =================
        moves = []
        visits = []

        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visits)

        if not visits:
            return None, None

        visits = np.array(visits, dtype=np.float32)

        # ================= TEMPERATURE =================
        temperature = self.temp_initial if move_number < self.temp_moves else 1e-6

        if temperature < 1e-6:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)

        return moves, probs



    # ------------------ HELPERS ------------------
    def _expand_batch(self, nodes):
        boards = torch.tensor(
            [encode_board(n.board) for n in nodes],
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            policy_logits, values = self.model(boards)

        for n, pl, v in zip(nodes, policy_logits, values):
            n.expand(pl.cpu())
            self._backpropagate(n, float(v.item()))

    def _expand(self, node: Node) -> float:
        board_tensor = torch.tensor(
            encode_board(node.board), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            policy_logits, value = self.model(board_tensor)

        value = float(value.item())
        node.expand(policy_logits[0].cpu())
        return value

    def _select(self, node: Node) -> Node:
        total_visits = sum(c.visits for c in node.children.values()) + 1
        best_score = -1e9
        best_child = None

        for child in node.children.values():
            q = child.value
            u = (
               self.cpuct
               * child.prior
               * math.sqrt(total_visits)
               / (1 + child.visits)
            )

            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backpropagate(self, node: Node, value: float):
        cur = node
        v = value
        while cur is not None:
            cur.visits += 1
            cur.value_sum += v
            v = -v
            cur = cur.parent

    def _add_dirichlet_noise(self, root: Node):
        if not root.children:
            return

        noise = torch.distributions.Dirichlet(
            torch.full((len(root.children),), self.dirichlet_alpha)
        ).sample()

        for child, n in zip(root.children.values(), noise):
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * float(n)
            )

    def _terminal_value(self, board: chess.Board) -> float:
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0
