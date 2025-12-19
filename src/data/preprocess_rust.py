import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
import re
import time
import random
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
import chess

from src.config import GM_GAMES_PATH, SHARD_DIR
from src.nn.encoding import encode_board, move_to_index, MOVE_SPACE

import rust_pgn


# ====== USER TUNABLES ======
SAMPLE_RATE = 0.02            # with 4.8M games, start 1–3%. 10% is massive.
CHUNK_SIZE = 50_000
SHARD_SIZE = 10_000           # safe; dense policy vectors are heavy
SAMPLE_EVERY_PLY = 1          # set to 2 or 3 to reduce dataset size
MAX_PLIES_PER_GAME = 240
MIN_PLIES_BEFORE_SAMPLING = 6
RANDOM_SEED = 42

RESUME_FROM_EXISTING_SHARDS = True
DEBUG_FIRST_N_PARSE_FAILURES = 10
# ===========================


TRAILING_NAG_RE = re.compile(r"([!?]+)$")
LEADING_MOVENUM_RE = re.compile(r"^(?:\d+\.{1,3}|\.{3})")

# Chess.com comment/annotation tokens
# Examples: "{[%clk", "0:05:00]}", "{[%eval", "0.34]}", "{", "}", "$1"
def is_noise_token(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return True

    # bracket/brace/comment fragments
    if t.startswith("{") or t.endswith("}") or t.startswith("}") or t == "{":
        return True
    if t.startswith("(") or t.endswith(")") or t == ")":
        return True

    # chess.com annotation payload pieces often look like "[%clk", "0:03:21]}", etc.
    if t.startswith("[%") or "%clk" in t or "%eval" in t or "%emt" in t:
        return True
    if "[" in t or "]" in t:
        # move SAN never contains [ or ] (those come from annotations)
        # Safe to treat as noise for chess.com PGNs
        return True

    # NAG numeric annotation
    if t.startswith("$") and t[1:].isdigit():
        return True

    # game termination markers that sometimes slip in
    if t in ("1-0", "0-1", "1/2-1/2", "*"):
        return True

    return False


def fmt_time(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def clean_san(tok: str) -> str:
    """
    Convert rust token stream -> parseable SAN token for python-chess.
    Handles:
      - chess.com annotation noise (filtered before calling this)
      - move numbers glued to tokens: "1.e4", "23...Nc6"
      - castling normalization
      - trailing ! ? sequences
    """
    tok = tok.strip()
    if not tok:
        return ""

    # Remove leading move number glued to SAN
    tok = LEADING_MOVENUM_RE.sub("", tok).strip()
    if not tok or tok == "...":
        return ""

    tok = tok.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    tok = TRAILING_NAG_RE.sub("", tok).strip()

    # Sometimes tokens like "e4+" are valid; keep them.
    return tok


def result_to_value_white(result_str: str) -> float:
    s = str(result_str).strip()
    if s == "1-0":
        return 1.0
    if s == "0-1":
        return -1.0
    if s == "1/2-1/2":
        return 0.0
    return 0.0


def safe_move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    try:
        return move_to_index(move)  # type: ignore
    except TypeError:
        try:
            return move_to_index(move, board)  # type: ignore
        except Exception:
            return None
    except Exception:
        return None


def find_starting_shard_id() -> int:
    os.makedirs(SHARD_DIR, exist_ok=True)
    existing = sorted(
        f for f in os.listdir(SHARD_DIR)
        if f.startswith("shard_") and f.endswith(".pt")
    )
    if not existing or not RESUME_FROM_EXISTING_SHARDS:
        return 1

    last = existing[-1]
    try:
        last_num = int(last.split("_")[1].split(".")[0])
        return last_num + 1
    except Exception:
        return 1


def flush_shard(shard_id: int, boards, policies, values, total_positions: int) -> int:
    shard_path = os.path.join(SHARD_DIR, f"shard_{shard_id:06d}.pt")
    log(f"Saving shard {shard_id} → {shard_path} (positions so far: {total_positions})")

    torch.save({
        "boards": torch.stack(boards),      # (N, 18, 8, 8)
        "policies": torch.stack(policies),  # (N, MOVE_SPACE)
        "values": torch.stack(values),      # (N, 1)
    }, shard_path)

    boards.clear()
    policies.clear()
    values.clear()
    return shard_id + 1


def preprocess():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.makedirs(SHARD_DIR, exist_ok=True)

    log(f"GM_GAMES_PATH = {GM_GAMES_PATH}")
    log(f"SHARD_DIR     = {SHARD_DIR}")
    log(f"SAMPLE_RATE   = {SAMPLE_RATE}")
    log(f"CHUNK_SIZE    = {CHUNK_SIZE}")
    log(f"SHARD_SIZE    = {SHARD_SIZE}")
    log(f"SAMPLE_EVERY  = {SAMPLE_EVERY_PLY}")
    log(f"MAX_PLIES     = {MAX_PLIES_PER_GAME}")
    log(f"MIN_PLIES     = {MIN_PLIES_BEFORE_SAMPLING}")

    shard_id = find_starting_shard_id()
    log(f"Starting shard_id={shard_id}")

    boards, policies, values = [], [], []

    total_positions = 0
    games_seen = 0
    games_sampled = 0
    games_used = 0
    games_skipped = 0

    parse_failures_printed = 0
    start_time = time.time()
    last_log_time = start_time

    # Robust CSV reading for huge file with many columns
    reader = pd.read_csv(
        GM_GAMES_PATH,
        usecols=lambda c: c.strip().lower() in {"pgn", "result"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        for pgn_text, result_str in chunk[["pgn", "result"]].itertuples(index=False, name=None):
            games_seen += 1

            if random.random() > SAMPLE_RATE:
                continue
            games_sampled += 1

            value_white = result_to_value_white(result_str)

            try:
                tokens = rust_pgn.extract_san_moves(pgn_text)
            except Exception:
                games_skipped += 1
                continue

            if not tokens:
                games_skipped += 1
                continue

            board = chess.Board()
            ply = 0
            ok = True
            positions_added = 0

            for raw_tok in tokens:
                if ply >= MAX_PLIES_PER_GAME:
                    break
                if board.is_game_over(claim_draw=True):
                    break

                if is_noise_token(raw_tok):
                    continue

                san = clean_san(raw_tok)
                if not san:
                    continue

                # Parse once
                try:
                    move = board.parse_san(san)
                except Exception:
                    ok = False
                    if parse_failures_printed < DEBUG_FIRST_N_PARSE_FAILURES:
                        parse_failures_printed += 1
                        log(f"[PARSE FAIL] token='{raw_tok}' cleaned='{san}' ply={ply} fen={board.fen()}")
                    break

                # Sample position before move
                if ply >= MIN_PLIES_BEFORE_SAMPLING and (ply % SAMPLE_EVERY_PLY == 0):
                    bt = torch.from_numpy(encode_board(board)).float()

                    policy = torch.zeros(MOVE_SPACE, dtype=torch.float32)
                    idx = safe_move_to_index(board, move)
                    if idx is not None:
                        policy[idx] = 1.0

                    # ✅ side-to-move target
                    z = value_white if board.turn == chess.WHITE else -value_white
                    vt = torch.tensor([z], dtype=torch.float32)

                    boards.append(bt)
                    policies.append(policy)
                    values.append(vt)

                    total_positions += 1
                    positions_added += 1

                    if len(boards) >= SHARD_SIZE:
                        shard_id = flush_shard(shard_id, boards, policies, values, total_positions)

                board.push(move)
                ply += 1

            if ok and positions_added > 0:
                games_used += 1
            else:
                games_skipped += 1

            now = time.time()
            if now - last_log_time >= 10.0:
                elapsed = now - start_time
                log(
                    f"chunk={chunk_idx} seen={games_seen} sampled={games_sampled} used={games_used} skipped={games_skipped} "
                    f"positions={total_positions} shards={shard_id-1} "
                    f"elapsed={fmt_time(elapsed)}"
                )
                last_log_time = now

    if boards:
        shard_id = flush_shard(shard_id, boards, policies, values, total_positions)

    elapsed = time.time() - start_time
    log("✔ Rust-assisted preprocessing complete.")
    log(f"Games seen:       {games_seen}")
    log(f"Games sampled:    {games_sampled}")
    log(f"Games used:       {games_used}")
    log(f"Games skipped:    {games_skipped}")
    log(f"Total positions:  {total_positions}")
    log(f"Total shards:     {shard_id - 1}")
    log(f"Total time:       {fmt_time(elapsed)}")


if __name__ == "__main__":
    preprocess()
