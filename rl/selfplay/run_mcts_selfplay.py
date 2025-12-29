from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import chess
from tqdm import tqdm

from rl.encoding import encode_planes
from rl.util import default_root, append_event, now_ms
from rl.mcts.puct import Node, outcome_value, select_child, expand_node, backup, policy_from_visits
from nn.models.chronos_hybrid import ChronosHybridNet


@torch.inference_mode()
def net_infer(model: ChronosHybridNet, x: torch.Tensor) -> tuple[np.ndarray, float]:
    out = model(x)
    pol = out["policy"].detach().cpu().numpy()[0].astype(np.float32)
    val = float(out["value"].detach().cpu().numpy()[0, 0])
    return pol, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Hybrid checkpoint .pt (kind=hybrid_policy)")
    ap.add_argument("--run", default="", help="Run id. Default az_<timestamp>")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--dir-alpha", type=float, default=0.3)
    ap.add_argument("--dir-frac", type=float, default=0.25)
    ap.add_argument("--temp", type=float, default=1.0, help="Move sampling temperature")
    ap.add_argument("--max-plies", type=int, default=240)
    args = ap.parse_args()

    root = default_root()
    run_id = args.run or f"az_{int(time())}"
    out_dir = root / "shards" / "az" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "az_selfplay.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ChronosHybridNet(in_planes=25).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rng = np.random.default_rng(1234)
    append_event(root, {"ts_ms": now_ms(), "type": "az_selfplay_start", "run_id": run_id, "games": args.games, "sims": args.sims})

    with out_jsonl.open("w", encoding="utf-8") as f:
        for gid in tqdm(range(1, args.games + 1), desc="AZ selfplay"):
            board = chess.Board()
            game_records: List[Dict[str, Any]] = []

            for ply in range(args.max_plies):
                term = outcome_value(board)
                if term is not None:
                    break

                root_node = Node(board_fen=board.fen(), to_play_white=(board.turn == chess.WHITE))

                planes = np.asarray(encode_planes(board), dtype=np.float32).reshape(25, 8, 8)
                x = torch.from_numpy(planes).unsqueeze(0).to(device)
                pri_logits, v = net_infer(model, x)

                expand_node(root_node, board, pri_logits, add_noise=True, rng=rng, noise_alpha=float(args.dir_alpha), noise_frac=float(args.dir_frac))
                for _ in range(int(args.sims)):
                    sim_board = chess.Board(root_node.board_fen)
                    node = root_node
                    path: List[Tuple[Node, chess.Move]] = []

                    # Selection
                    while node.expanded and not node.is_terminal():
                        mv = select_child(node, float(args.cpuct))
                        path.append((node, mv))
                        sim_board.push(mv)

                        child = node.children.get(mv)
                        if child is None:
                            child = Node(board_fen=sim_board.fen(), to_play_white=(sim_board.turn == chess.WHITE))
                            node.children[mv] = child
                        node = child

                    # Evaluation / Expansion
                    leaf_term = outcome_value(sim_board)
                    if leaf_term is not None:
                        node.terminal_value = float(leaf_term)
                        leaf_value = float(leaf_term)
                    else:
                        planes2 = np.asarray(encode_planes(sim_board), dtype=np.float32).reshape(25, 8, 8)
                        x2 = torch.from_numpy(planes2).unsqueeze(0).to(device)
                        pri2, leaf_value = net_infer(model, x2)
                        # Expand leaf without noise (noise only at root)
                        expand_node(node, sim_board, pri2, add_noise=False, rng=rng, noise_alpha=float(args.dir_alpha), noise_frac=float(args.dir_frac))

                    backup(path, float(leaf_value))

                idxs, ps = policy_from_visits(root_node)

                # choose move from visit distribution
                moves = list(root_node.edges.keys())
                visits = np.array([root_node.edges[mv].n for mv in moves], dtype=np.float32)
                if visits.sum() <= 0:
                    probs = np.ones_like(visits) / len(visits)
                else:
                    if args.temp <= 1e-6:
                        probs = np.zeros_like(visits)
                        probs[int(np.argmax(visits))] = 1.0
                    else:
                        xw = np.power(visits, 1.0 / float(args.temp))
                        probs = xw / max(1e-9, xw.sum())

                mv = moves[int(rng.choice(len(moves), p=probs))]

                game_records.append({
                    "run_id": run_id,
                    "game_id": gid,
                    "ply": ply,
                    "fen": board.fen(),
                    "planes": encode_planes(board),
                    "policy": {"idx": idxs, "p": ps},
                    "value_bootstrap": float(v),
                })

                board.push(mv)

            res = board.result(claim_draw=True)
            if res == "1-0":
                white_out = 1.0
            elif res == "0-1":
                white_out = -1.0
            else:
                white_out = 0.0

            for rec in game_records:
                b = chess.Board(rec["fen"])
                pov = white_out if b.turn == chess.WHITE else -white_out
                rec["result"] = res
                rec["value_target"] = float(pov)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    append_event(root, {"ts_ms": now_ms(), "type": "az_selfplay_done", "run_id": run_id, "out": str(out_jsonl)})
    print(f"Done: {out_jsonl}")


if __name__ == "__main__":
    main()
