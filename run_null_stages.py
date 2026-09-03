"""Run the four rotation inference stages in sequence, logging each exit code."""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
for sfx in ("_v4", "_disap"):
    for flags in ([], ["--ipbridge"]):
        label = f"{sfx} {'ipbridge' if flags else 'decoy'}"
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", "db_analysis_identity_rotation.py",
                            "--roster", "outputs/roster_v2.csv", "--suffix", sfx] + flags,
                           cwd=HERE, capture_output=True, text=True)
        print(f"{label}: exit={r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
        if r.returncode != 0:
            print((r.stderr or "")[-1200:], flush=True)
print("ALL DONE", flush=True)
