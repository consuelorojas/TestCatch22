#!/usr/bin/env python3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of sweep scripts
'''
SCRIPTS = [
#    "sweeps/fhn/bparam_fhn.py",
#    "sweeps/fhn/periods_fhn.py",
#    "sweeps/fhn/npoints_fhn.py",
#    "sweeps/fhn/periods_fhn.py",
#    "sweeps/fhn/samples_fhn.py",
#    "sweeps/fhn/noise_fhn.py",
#    "sweeps/fhn_obs/bparam.py",
#    "sweeps/fhn_obs/periods.py",
#    "sweeps/fhn_obs/npoints.py",
#    "sweeps/fhn_obs/samples.py",
#    "sweeps/fhn_obs/noise.py",
#    "sweeps/sinusoidal/periods_sine.py",
#    "sweeps/sinusoidal/npoints_sine.py",
#    "sweeps/sinusoidal/samples_sine.py",
#    "sweeps/sinusoidal/frequency_sine.py"
]

SCRIPTS = [
    'notebooks/p_values.py'
]
'''
SCRIPTS = [
    "sweeps/fhn/time_sample_fhn.py",
    "sweeps/fhn/time_npoints.py",
    "sweeps/fhn/time_periods.py",
    "sweeps/fhn_obs/times_npoints.py",
    "sweeps/fhn_obs/times_periods.py",
    "sweeps/fhn_obs/times_samples.py",
    "sweeps/sinusoidal/times_npoints.py",
    "sweeps/sinusoidal/times_periods.py",
    "sweeps/sinusoidal/times_sine.py"
]

# Number of parallel workers (default: run all at once)

MAX_WORKERS = 999

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
