"""Wait for the null warm, then run the remaining v4 stages serially."""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path(r"")
PS = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "Where-Object { $_.CommandLine -like '*warm_null_windows*' }).Count")


def procs():
    """-1 == unknown. An unreadable count must never greenlight; it can only veto."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return -1
    t = (out.stdout or "").strip()
    return int(t) if out.returncode == 0 and t.isdigit() else -1


def warm_done():
    try:
        return "done in" in LOG.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


print("[v4] waiting for the null warm", flush=True)
q = 0
while True:
    if warm_done() and procs() <= 0:
        q += 1
        print(f"[v4] warm looks done ({q}/2)", flush=True)
        if q >= 2:
            break
    else:
        q = 0
    time.sleep(30)

for label, flags in [("STAGE 2 IP bridge", ["--ipbridge"]),
                     ("STAGE 3 placebo", ["--placebo"]),
                     ("STAGE 4 planted control", ["--selftest"])]:
    print(f"\n[v4] === {label} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", "db_analysis_identity_rotation.py",
                        "--roster", "outputs/roster_v2.csv", "--suffix", "_v4"] + flags, cwd=HERE)
    print(f"[v4] {label} exit={r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
print("\n[v4] === ALL DONE ===", flush=True)
