#!/usr/bin/env python3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of sweep scripts
SCRIPTS = [
    "sweeps/fhn/samples_fhn.py",
    "sweeps/fhn_obs/samples.py",
    "sweeps/sinusoidal/samples_sine.py"
]

# Number of parallel workers (default: run all at once)
MAX_WORKERS = 2

def run_script(script):
    print(f"→ Running {script}")
    rc = subprocess.call(["python3", script])
    if rc == 0:
        print(f"✅ Finished {script}")
    else:
        print(f"❌ {script} failed (exit code {rc})")
    return (script, rc)

def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_script, s) for s in SCRIPTS]
        for future in as_completed(futures):
            script, rc = future.result()
    print("\nAll scripts finished.")

if __name__ == "__main__":
    main()
