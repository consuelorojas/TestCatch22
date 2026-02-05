import os
import sys
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm

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
noise = 0.1

samples = np.arange(10, 520, 10)

## Output directory
sweep_name = "sine/sine_samples"
output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"results_{timestamp}.pkl")

# Run sweep
all_results = []
for i, s in enumerate(tqdm(samples, desc="Sweeping number of samples")):
    X, y = create_labeled_dataset( #type: ignore
        [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
         (1, 'sine', {'args': [f1, noise, npoints, nperiods]})],
        n_samples_per_class=s
    )
    splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
    results = run_experiment(X, y, splits)

    all_results.append({
        'samples': s,
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")