"""Run a command and write its combined stdout/stderr to a log file with per-line timestamps.

Used by Streamlit local_control so logs have timestamps even for warnings and external libs.

Usage:
  python -u python/streamlit/common/run_with_ts.py --log <path> -- <cmd...>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to a log file. Output will be appended.")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (prefix with --)")
    args = ap.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("ERROR: no command provided. Use: -- <cmd...>", file=sys.stderr)
        return 2

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"[{_ts()}] --- RUN {cmd} ---\n")
        lf.flush()

        # NOTE: text=False (bytes) so we can treat '\r' as line breaks (progress bars).
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=os.environ.copy(),
        )

        buf = ""
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            # Treat carriage return as newline so progress bars don't clobber log lines.
            piece = chunk.decode("utf-8", errors="replace").replace("\r", "\n")
            buf += piece
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line == "":
                    continue
                lf.write(f"[{_ts()}] {line}\n")
                lf.flush()

        # Flush remainder
        rem = buf.strip("\n")
        if rem:
            lf.write(f"[{_ts()}] {rem}\n")

        rc = proc.wait()
        lf.write(f"[{_ts()}] --- EXIT rc={rc} ---\n")
        lf.flush()

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
