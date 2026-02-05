import os
import sys
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits


## sweep configuration
fbase = 5
f1 = 5.18
#npoints = [1, 2, 3, 5, 7, 10, 15, 20, 30]
npoints = np.arange(1, 21)  # From 1 to 20 in steps of 2
nperiods = 3

noise = 0.1
samples = 100

## Output directory
sweep_name = "sine/sine_points"
output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"results_{timestamp}.pkl")

# Run sweep
all_results = []
for i, npp in enumerate(tqdm(npoints, desc="Sweeping number of points")):
    X, y = create_labeled_dataset( # type: ignore
        [(0, 'sine', {'args': [fbase, 0.1, npp, nperiods]}),
         (1, 'sine', {'args': [f1, 0.1, npp, nperiods]})],
        n_samples_per_class=samples
    )
    splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
    results = run_experiment(X, y, splits)

    all_results.append({
        'npp': npp,
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")