import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import time_experiment
from dataset import create_labeled_dataset, get_kfold_splits

## sweep configuration
b0 = 0.1

b1 = 1.0
db_obs = 0.032

db1 = 0.178
b12 = b1 + db1
dt = 0.1


# step to subsampling
pseudo_period = 30
npp = 10          # change the number of points per period here!
step = int(pseudo_period / npp / dt)

epsilon = 0.2
I = 0
noise = 0.1

trans = 100 # transient

samples = np.arange(10, 520, 10)
#samples = [10]

all_results = []
for sample in tqdm(samples):
    X, y, t = create_labeled_dataset([
        (0, 'fhn', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
        (1, 'fhn', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, noise]})],
        n_samples_per_class=sample, subsample_step = step, transient = trans, return_time=True
        )

    splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
    results = time_experiment(X, y, splits)
    all_results.append({
        'sample': sample,
        'results': results
    })

# Save results
output_dir = os.path.join("notebooks", "times", "fhn_samples")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "fhn_dyn_samples_times_results.pkl")
with open(output_file, "wb") as f:
    pickle.dump(all_results, f)
print(f"FHN dyn Sweep complete. Results saved to {output_file}")


### FHN obs
all_fhn_results = []
for i, s in enumerate(tqdm(samples, desc="Sweeping number of samples")):
    X, y, t = create_labeled_dataset([
        (0, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
        (1, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1+db_obs, epsilon, I, noise]})],
        n_samples_per_class=s, subsample_step = step, transient = trans, return_time=True
        )
    splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
    results = time_experiment(X, y, splits)
    all_fhn_results.append({
        'samples': s,
        'results': results
    })
# Save FHN results
output_file_fhn = os.path.join(output_dir, "fhn_obs_samples_times_results.pkl")
with open(output_file_fhn, "wb") as f:
    pickle.dump(all_fhn_results, f)
print(f"FHN Obs Sweep complete. Results saved to {output_file_fhn}")









### SINE WAVES

## sweep configuration
fbase = 5
f1 = 5.18
nperiods = 3
npoints = 7


all_sine_results = []
for i, s in enumerate(tqdm(samples, desc="Sweeping number of samples")):
    X, y = create_labeled_dataset(
        [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
         (1, 'sine', {'args': [f1, noise, npoints, nperiods]})],
        n_samples_per_class=s
    )
    splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
    results = time_experiment(X, y, splits)
    all_sine_results.append({
        'samples': s,
        'results': results
    })

# Save sine wave results
output_file_sine = os.path.join(output_dir, "sine_samples_times_results.pkl")
with open(output_file_sine, "wb") as f:
    pickle.dump(all_sine_results, f)
print(f"Sine Sweep complete. Results saved to {output_file_sine}")
