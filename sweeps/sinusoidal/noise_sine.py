import os
import sys
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits


## sweep configuration
fbase = 5
f1 = 5.18
nperiods = 3
npoints = 7

#noise = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
noise = np.linspace(0, 0.4, 20)
bignoise = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
noise = np.concatenate((noise, bignoise))
noise = np.round(noise, 3)

samples = 100

## Output directory
sweep_name = "sine/sine_noise"
output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"results_{timestamp}.pkl")

# Run sweep

# worker function

def run_single_experiment(n):
    X, y = create_labeled_dataset( #type:ignore
        [(0, 'sine', {'args': [fbase, n, npoints, nperiods]}),
         (1, 'sine', {'args': [f1, n, npoints, nperiods]})],
        n_samples_per_class=samples
    )
    splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
    results = run_experiment(X, y, splits, ffts=True)

    return {
        'noise': n,
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca'],
        'fft': results['fft'],
        'fft_pca': results['fft_pca']
    }

def main():
    all_results = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_single_experiment, n): n for n in noise
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Sweeping noise levels"
        ):
            all_results.append(future.result())

    # save results
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Sweep complete. Results saved to {output_file}")