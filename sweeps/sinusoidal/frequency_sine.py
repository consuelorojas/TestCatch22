import os
import sys
import pickle
from datetime import datetime

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits


## sweep configuration
fbase = 5
deltaf = 0.1
f1 = [fbase + deltaf*i for i in range(0, 15, 1)]
dfreq = [deltaf *  i for i in range(0, 15, 1)]

npoints = 5
nperiods = 3

noise = 0.1
samples = 100

## Output directory
sweep_name = "sine_frequency"
output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"results_{timestamp}.pkl")

# Run sweep
all_results = []
for i, freq in enumerate(f1):
    X, y = create_labeled_dataset(
        [(0, 'sine', {'args': [fbase, 0.1, npoints, nperiods]}),
         (1, 'sine', {'args': [freq, 0.1, npoints, nperiods]})],
        n_samples_per_class=samples
    )
    splits = get_kfold_splits(X, y, n_splits=5, stratified=False)
    results = run_experiment(X, y, splits)

    all_results.append({
        'df': round(freq - fbase, 3),
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")