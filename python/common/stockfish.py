from __future__ import annotations
import subprocess
import re
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional

_SCORE_RE = re.compile(r"score (cp|mate) (-?\d+)")


@dataclass
class SfEval:
    cp: int
    depth: int
    mate: Optional[int] = None


class Stockfish:
    """
    Robust UCI wrapper with a background reader thread so we can enforce timeouts.

    This avoids rare hangs where blocking iteration over stdout can stall forever.
    """

    def __init__(self, exe_path: str):
        self.exe_path = exe_path
        self._q: "queue.Queue[str]" = queue.Queue()
        self._last_lines: list[str] = []
        self._stop = threading.Event()

        self.p = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if not self.p.stdin or not self.p.stdout:
            raise RuntimeError("Failed to open Stockfish pipes")

        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

        # UCI handshake
        self._send("uci")
        self._drain_until("uciok", timeout_s=10.0)
        self._send("isready")
        self._drain_until("readyok", timeout_s=10.0)

    def _reader(self):
        assert self.p.stdout
        try:
            while not self._stop.is_set():
                line = self.p.stdout.readline()
                if line == "":
                    break  # EOF
                line = line.rstrip("\r\n")
                # Keep a small tail for debugging
                self._last_lines.append(line)
                if len(self._last_lines) > 50:
                    self._last_lines = self._last_lines[-50:]
                self._q.put(line)
        except Exception:
            pass

    def quit(self):
        try:
            self._send("quit")
        except Exception:
            pass
        self._stop.set()
        try:
            self.p.kill()
        except Exception:
            pass

    def setoption(self, name: str, value: str):
        self._send(f"setoption name {name} value {value}")

    def eval_fen_depth(self, fen: str, depth: int, timeout_s: float = 15.0) -> SfEval:
        """
        Evaluate a position at fixed depth. Times out if Stockfish doesn't return bestmove.
        """
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        best_cp = 0
        best_depth = 0
        best_mate: Optional[int] = None

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            if line.startswith("info"):
                m = _SCORE_RE.search(line)
                if m:
                    kind = m.group(1)
                    val = int(m.group(2))
                    md = re.search(r"depth (\d+)", line)
                    d = int(md.group(1)) if md else best_depth
                    if kind == "cp":
                        best_cp = val
                        best_depth = d
                        best_mate = None
                    else:
                        best_mate = val
                        sign = 1 if val > 0 else -1
                        best_cp = sign * (32000 - 100 * min(abs(val), 320))
                        best_depth = d
            elif line.startswith("bestmove"):
                return SfEval(cp=best_cp, depth=best_depth, mate=best_mate)

        raise TimeoutError("Stockfish eval timeout. Last lines: " + " | ".join(self._last_lines[-10:]))

    def _send(self, cmd: str):
        assert self.p.stdin
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _drain_until(self, token: str, timeout_s: float = 5.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if token in line:
                return
        raise TimeoutError(f"Timeout waiting for '{token}'. Last lines: " + " | ".join(self._last_lines[-10:]))
