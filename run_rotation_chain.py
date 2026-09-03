"""Wait for the cache warms to finish, then run the four rotation stages in order.

Written as Python rather than shell because the shell version got the wait condition wrong:
`grep -c ... || echo 0` emits two values when the file is missing, and `[ "$a" -ge 1 ]` then
fails with "integer expression expected". A wait condition that errors is worse than no wait
condition, because the stages would start against a half-warmed cache and silently under-cover.

The wait is on *process liveness*, not on log text: any warm process still running means the
cache is still filling. That survives a lost wrapper, a redirect that never flushed, and a warm
that dies without writing its summary line.

Usage: python run_rotation_chain.py [--suffix _v3] [--roster outputs/roster_v2.csv]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

PS_COUNT = (
    "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "Where-Object { $_.CommandLine -like '*warm_agent_cache*' }).Count"
)


def warms_running() -> int:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", PS_COUNT],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0)
    except Exception:
        return -1          # unknown: treat as still running, never as finished


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--roster", default="outputs/roster_v2.csv")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()

    print("[chain] waiting for cache warms to finish", flush=True)
    quiet = 0
    while True:
        n = warms_running()
        if n == 0:
            quiet += 1
            # two consecutive quiet polls, so a momentary miscount cannot start the stages early
            if quiet >= 2:
                break
        else:
            quiet = 0
        time.sleep(a.poll)
    print("[chain] warms finished; starting rotation stages", flush=True)

    stages = [
        ("STAGE 1: decoy/honeypot bridge", []),
        ("STAGE 2: IP bridge", ["--ipbridge"]),
        ("STAGE 3: placebo", ["--placebo"]),
        ("STAGE 4: planted-handoff positive control", ["--selftest"]),
    ]
    for label, flags in stages:
        print(f"\n[chain] === {label} ===", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-u", "db_analysis_identity_rotation.py",
             "--roster", a.roster, "--suffix", a.suffix] + flags,
            cwd=HERE,
        )
        print(f"[chain] {label} exit={r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
        if r.returncode != 0:
            # Keep going: a failed stage is recorded and the later stages are independent
            # enough to be worth having. The exit code is in the log for the morning.
            print(f"[chain] WARNING: {label} returned {r.returncode}", flush=True)
    print("\n[chain] === ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
