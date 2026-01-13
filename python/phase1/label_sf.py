from __future__ import annotations
# --- Chronos bootstrap: ensure ./python is on sys.path (works with Streamlit & direct script runs)
import sys
from pathlib import Path
_p = Path(__file__).resolve()
for _ in range(10):
    if _p.name.lower() == "python":
        break
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
# --- end bootstrap

import argparse
import gzip
import json
import threading
import time as _time
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
from queue import Empty as QueueEmpty, Full as QueueFull

from common.paths import ensure_layout
from common.stockfish import Stockfish
from common.logging_utils import log_line
from common.public_export import export_public_status


def worker_main(worker_id: int, sf_path: str, depth: int, in_q: Queue, out_q: Queue, ready_q: Queue):
    try:
        sf = Stockfish(sf_path)
        sf.setoption("Threads", "1")
        sf.setoption("Hash", "256")
        sf.setoption("MultiPV", "1")
        sf.setoption("UCI_AnalyseMode", "true")
        sf._send("isready")
        sf._drain_until("readyok", timeout_s=10.0)
        ready_q.put((worker_id, "READY"))
    except Exception as e:
        try:
            ready_q.put((worker_id, f"FAIL:{type(e).__name__}:{e}"))
        except Exception:
            pass
        return

    while True:
        item = in_q.get()
        if item is None:
            break
        idx, fen = item
        try:
            ev = sf.eval_fen_depth(fen, depth, timeout_s=20.0)
            out_q.put((idx, fen, ev.cp, ev.depth, ev.mate))
        except Exception:
            out_q.put((idx, fen, None, None, None))
    sf.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_fens", type=str, required=True, help="Text file: 1 FEN per line")
    ap.add_argument("--out_jsonl_gz", type=str, required=True, help="Output labeled jsonl.gz")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, max(1, cpu_count() - 2)))
    ap.add_argument("--stockfish", type=str, default="", help="Path to stockfish.exe (default: E:\\chronos\\private\\bin\\stockfish.exe)")
    ap.add_argument("--queue_max", type=int, default=8192)
    args = ap.parse_args()

    p = ensure_layout()
    log_path = p["logs"] / "label_sf.log"
    log_line(log_path, f"START label_sf depth={args.depth} workers={args.workers}")

    sf_path = args.stockfish.strip() or str((p["bin"] / "stockfish.exe"))
    if not Path(sf_path).exists():
        log_line(log_path, f"ERROR Stockfish not found: {sf_path}")
        raise FileNotFoundError(f"Stockfish not found: {sf_path}")

    in_path = Path(args.in_fens)
    out_path = Path(args.out_jsonl_gz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Count lines (avoid loading all into RAM)
    n = 0
    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1

    log_line(log_path, f"Loaded {n} FENs from {in_path}")
    if n == 0:
        log_line(log_path, "ERROR No FENs found (input file is empty).")
        export_public_status({"phase": "IDLE", "last": "label_sf", "error": "empty_input"})
        return

    in_q: Queue = Queue(maxsize=args.queue_max)
    out_q: Queue = Queue(maxsize=args.queue_max)
    ready_q: Queue = Queue(maxsize=args.workers * 2)

    procs: list[Process] = []
    for i in range(args.workers):
        pr = Process(target=worker_main, args=(i, sf_path, args.depth, in_q, out_q, ready_q), daemon=True)
        pr.start()
        procs.append(pr)

    # Wait for READY from all workers (or FAIL), so we never silently hang.
    ready = {}
    deadline = _time.time() + 30.0
    while _time.time() < deadline and len(ready) < args.workers:
        try:
            wid, msg = ready_q.get(timeout=1.0)
            ready[wid] = msg
            log_line(log_path, f"Worker {wid}: {msg}")
        except QueueEmpty:
            pass

    failed = {k: v for k, v in ready.items() if not str(v).startswith("READY")}
    if len(ready) < args.workers or failed:
        log_line(log_path, f"ERROR Worker init failed. ready={len(ready)}/{args.workers} failed={failed}")
        export_public_status({"phase": "IDLE", "last": "label_sf", "error": "worker_init_failed", "details": failed, "ready": len(ready)})
        return

    status = {
        "phase": "SL_LABELING",
        "stage": "run",
        "depth": args.depth,
        "workers": args.workers,
        "total": n,
        "enqueued": 0,
        "done": 0,
    }
    export_public_status(status)

    # --- Writer thread: drain out_q continuously to avoid deadlock (out_q filling blocks workers) ---
    done_lock = threading.Lock()
    done = 0
    stop_evt = threading.Event()

    def writer_thread():
        nonlocal done
        last_log = 0.0
        with gzip.open(out_path, "wt", encoding="utf-8") as gz:
            while not stop_evt.is_set():
                with done_lock:
                    if done >= n:
                        break
                try:
                    idx, fen, cp, used_depth, mate = out_q.get(timeout=1.0)
                except QueueEmpty:
                    # If nobody is producing anymore, avoid waiting forever
                    alive = [i for i, pr in enumerate(procs) if pr.is_alive()]
                    with done_lock:
                        d = done
                    now = _time.time()
                    if now - last_log > 5.0:
                        log_line(log_path, f"Waiting... alive_workers={alive} done={d}/{n} enqueued={status['enqueued']}/{n}")
                        last_log = now
                    if len(alive) == 0 and d < n:
                        log_line(log_path, "ERROR All workers died during labeling. Aborting.")
                        export_public_status({"phase": "IDLE", "last": "label_sf", "error": "all_workers_dead", "done": d, "total": n})
                        stop_evt.set()
                        break
                    continue

                rec = {"idx": idx, "fen": fen, "cp": cp, "depth": used_depth, "mate": mate}
                gz.write(json.dumps(rec) + "\n")

                with done_lock:
                    done += 1
                    d = done

                if d % 2000 == 0 or d == n:
                    status["done"] = d
                    export_public_status(status)
                    log_line(log_path, f"Labeled {d}/{n} positions...")
                    try:
                        gz.flush()
                    except Exception:
                        pass

    wt = threading.Thread(target=writer_thread, daemon=True)
    wt.start()

    # --- Enqueue streaming (main thread) ---
    enq = 0
    last_full_log = 0.0
    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            fen = line.strip()
            if not fen:
                continue

            while True:
                try:
                    in_q.put((enq, fen), timeout=1.0)
                    break
                except QueueFull:
                    now = _time.time()
                    if now - last_full_log > 5.0:
                        alive = [i for i, pr in enumerate(procs) if pr.is_alive()]
                        with done_lock:
                            d = done
                        log_line(log_path, f"Queue full at enqueued={enq}/{n} done={d}/{n}. alive_workers={alive}")
                        last_full_log = now
                        if len(alive) == 0:
                            log_line(log_path, "ERROR All workers dead while enqueuing. Aborting.")
                            export_public_status({"phase": "IDLE", "last": "label_sf", "error": "all_workers_dead_enqueue", "enqueued": enq, "done": d, "total": n})
                            stop_evt.set()
                            wt.join(timeout=2)
                            return

            enq += 1
            if enq % 2000 == 0:
                status["enqueued"] = enq
                export_public_status(status)
                log_line(log_path, f"Enqueued {enq}/{n} FENs...")

    # Tell workers to stop after queue drains
    for _ in procs:
        in_q.put(None)

    log_line(log_path, "Enqueue complete. Waiting for writer to finish...")
    status["enqueued"] = n
    export_public_status(status)

    # Wait for writer to finish
    while True:
        with done_lock:
            d = done
        if d >= n:
            break
        _time.sleep(1.0)

    stop_evt.set()
    wt.join(timeout=5)

    for pr in procs:
        pr.join(timeout=2)

    log_line(log_path, f"DONE label_sf out={out_path} n={n}")
    export_public_status({"phase": "IDLE", "last": "label_sf", "out": str(out_path), "n": n})


if __name__ == "__main__":
    main()
