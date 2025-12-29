from __future__ import annotations
import argparse, csv, os, re
from pathlib import Path
from typing import Dict, List, Tuple

def norm(s: str) -> str:
    return (s or "").strip().lower()

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name) or "Player"

def load_players(players_file: Path) -> Dict[str, Tuple[str, set]]:
    """
    returns: display -> (outfile, set(usernames_lower))
    """
    out = {}
    for line in players_file.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("|") if p.strip()]
        display = parts[0]
        usernames = set(norm(p) for p in parts[1:])
        if not usernames:
            continue
        outname = safe_filename(display) + ".pgn"
        out[display] = (outname, usernames)
    return out

def to_int(x: str):
    try:
        x = (x or "").strip()
        return int(float(x)) if x else None
    except:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="gm_games.csv")
    ap.add_argument("--players-file", required=True)
    ap.add_argument("--outdir", default="", help="default: E:/chronos/datasets/pgn/players")
    ap.add_argument("--player-col", default="player")
    ap.add_argument("--pgn-col", default="pgn")
    ap.add_argument("--min-player-rating", type=int, default=0, help="uses player_rating column if present")
    ap.add_argument("--player-rating-col", default="player_rating")
    ap.add_argument("--progress-every", type=int, default=200000)
    args = ap.parse_args()

    inp = Path(args.input)
    root = os.environ.get("CHRONOS_DATA_ROOT", r"E:\chronos")
    outdir = Path(args.outdir) if args.outdir else (Path(root) / "datasets" / "pgn" / "players")
    outdir.mkdir(parents=True, exist_ok=True)

    players = load_players(Path(args.players_file))

    # reverse map: username -> display
    user_to_display: Dict[str, List[str]] = {}
    for display, (_, users) in players.items():
        for u in users:
            user_to_display.setdefault(u, []).append(display)

    # open handles on demand
    handles = {}
    counts = {display: 0 for display in players}
    total = matched = 0

    with inp.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            u = norm(row.get(args.player_col, ""))
            if not u:
                continue

            displays = user_to_display.get(u)
            if not displays:
                continue

            if args.min_player_rating > 0:
                pr = to_int(row.get(args.player_rating_col, ""))
                if pr is None or pr < args.min_player_rating:
                    continue

            pgn = (row.get(args.pgn_col, "") or "").strip()
            if not pgn:
                continue

            matched += 1
            txt = pgn.rstrip() + "\n\n"

            for display in displays:
                outname, _ = players[display]
                if outname not in handles:
                    handles[outname] = (outdir / outname).open("a", encoding="utf-8", newline="")
                handles[outname].write(txt)
                counts[display] += 1

            if args.progress_every and total % args.progress_every == 0:
                print(f"[progress] rows={total:,} matched={matched:,}")

    for fh in handles.values():
        try:
            fh.flush(); fh.close()
        except:
            pass

    print("\nDone.")
    print(f"Total rows scanned: {total:,}")
    print(f"Rows matched: {matched:,}")
    for display, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        outname, _ = players[display]
        print(f"  {display}: {c:,} -> {outname}")
    print(f"\nOutput directory: {outdir}")

if __name__ == "__main__":
    main()
