import os
from collections import Counter
import chess.pgn
from datetime import datetime

# ===================== CONFIG (NO ARGS NEEDED) =====================
PGN_DIR = os.path.join("data", "PGN")
OPENING_PLIES = 8             # first N plies for opening diversity
MIN_GAMES_WARNING = 10        # warn if too few games
# ==================================================================


def find_pgn_files(pgn_dir):
    if not os.path.exists(pgn_dir):
        return []
    return [
        os.path.join(pgn_dir, f)
        for f in os.listdir(pgn_dir)
        if f.lower().endswith(".pgn")
    ]


def load_games(pgn_files):
    games = []
    for path in pgn_files:
        with open(path, encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
    return games


def extract_opening(game, plies):
    board = game.board()
    moves = []
    for i, move in enumerate(game.mainline_moves()):
        if i >= plies:
            break
        moves.append(move.uci())
        board.push(move)
    return tuple(moves)


def analyze_games(games):
    openings = Counter()
    results = Counter()
    colors = Counter()
    lengths = []

    for g in games:
        openings[extract_opening(g, OPENING_PLIES)] += 1
        results[g.headers.get("Result", "*")] += 1
        colors[g.headers.get("White", "White")] += 1
        lengths.append(len(list(g.mainline_moves())))

    return openings, results, colors, lengths


def print_report(games, openings, results, colors, lengths):
    print("\n================ DIVERSITY TEST REPORT ================")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Games analyzed: {len(games)}")

    if len(games) < MIN_GAMES_WARNING:
        print("⚠️ WARNING: Too few games for reliable diversity conclusions")

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n--- Color Distribution (White names) ---")
    for k, v in colors.most_common():
        print(f"{k}: {v}")

    print("\n--- Game Length ---")
    print(f"Average plies: {sum(lengths)/len(lengths):.1f}")
    print(f"Min plies: {min(lengths)}")
    print(f"Max plies: {max(lengths)}")

    print("\n--- Opening Diversity ---")
    unique_openings = len(openings)
    most_common = openings.most_common(5)

    print(f"Unique openings (first {OPENING_PLIES} plies): {unique_openings}")
    print("Top repeated openings:")
    for seq, cnt in most_common:
        print(f"{cnt:4d}x {' '.join(seq)}")

    if most_common and most_common[0][1] / len(games) > 0.5:
        print("\n❌ COLLAPSE WARNING: >50% games share the same opening")
    else:
        print("\n✅ Opening diversity looks healthy")

    print("=======================================================\n")


def main():
    print("[INFO] Running diversity test (no arguments required)")

    pgn_files = find_pgn_files(PGN_DIR)
    if not pgn_files:
        print("❌ No PGN files found.")
        print(f"Expected directory: {PGN_DIR}")
        return

    print(f"[INFO] Found {len(pgn_files)} PGN file(s)")
    games = load_games(pgn_files)

    if not games:
        print("❌ PGN files found, but no games inside.")
        return

    openings, results, colors, lengths = analyze_games(games)
    print_report(games, openings, results, colors, lengths)


if __name__ == "__main__":
    main()
