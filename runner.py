#!/usr/bin/env python3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of sweep scripts
SCRIPTS = [
    #"sweeps/fhn/bparam_fhn.py",
    #"sweeps/fhn/periods_fhn.py",
    #"sweeps/fhn/npoints_fhn.py",
    #"sweeps/fhn/periods_fhn.py",
    #"sweeps/fhn/samples_fhn.py",
    #"sweeps/fhn_obs/bparam.py",
    #"sweeps/fhn_obs/periods.py",
    #"sweeps/fhn_obs/npoints.py",
    #"sweeps/fhn_obs/samples.py",
    #"sweeps/sinusoidal/periods_sine.py",
    #"sweeps/sinusoidal/npoints_sine.py",
    #"sweeps/sinusoidal/samples_sine.py"
]


"""
SCRIPTS = [
    "results/fhn/fhn_noise/plot_sweep_results_noise.py",
    "results/fhn/fhn_npp/plot_sweep_results_points.py",
    "results/fhn/fhn_parameter/plot_sweep_results_param.py",
    "results/fhn/fhn_periods/plot_sweep_results_periods.py",
    "results/fhn/fhn_samples/plot_sweep_results_samples.py",
    "results/fhn_obs/noise/plot_sweep_results_noise.py",
    "results/fhn_obs/npoints/plot_sweep_results_points.py",
    "results/fhn_obs/parameter/plot_sweep_results_param.py",
    "results/fhn_obs/periods/plot_sweep_results_periods.py",
    "results/fhn_obs/samples/plot_sweep_results_samples.py",
    "results/sine/sine_frequency/plot_sweep_results_freq.py",
    "results/sine/sine_noise/plot_sweep_results_noise.py",
    "results/sine/sine_periods/plot_sweep_results_periods.py",
    "results/sine/sine_points/plot_sweep_results_points.py",
    "results/sine/sine_samples/plot_sweep_results_param.py"

]
"""

# Number of parallel workers (default: run all at once)

MAX_WORKERS = 10

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
