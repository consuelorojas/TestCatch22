import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import time_experiment
from dataset import create_labeled_dataset, get_kfold_splits

fbase = 1
df = 0.18
f1 = fbase + df

npoints = 7
nperiods = 3

noise = 0.1

samples = 100

X,y = create_labeled_dataset(
    [(0, 'sine', {'args': [fbase, 0.1, npoints, nperiods]}),
     (1, 'sine', {'args': [f1, 0.1, npoints, nperiods]})],
    n_samples_per_class=samples
)

splits = get_kfold_splits(X, y, n_splits=5, stratified=False)
results = time_experiment(X, y, splits)

print("Timing Results (seconds):")
print(f"Raw: {np.mean(results['raw']):.4f} ± {np.std(results['raw']):.4f}")
print(f"Raw + PCA: {np.mean(results['pca']):.4f} ± {np.std(results['pca']):.4f}")
print(f"Features: {np.mean(results['features']):.4f} ± {np.std(results['features']):.4f}")
print(f"Features + PCA: {np.mean(results['features_pca']):.4f} ± {np.std(results['features_pca']):.4f}")