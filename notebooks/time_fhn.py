import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import time_experiment
from dataset import create_labeled_dataset, get_kfold_splits

## sweep configuration
b0 = 0.1

b1 = 1.0
db1 = 0.032
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

samples = 80


X, y, t = create_labeled_dataset([
    (0, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
    (1, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, noise]})],
    n_samples_per_class=samples, subsample_step = step, transient = trans, return_time=True
    )

splits = get_kfold_splits(X, y, n_splits=5, stratified=False)
results = time_experiment(X, y, splits)

print("Timing Results (seconds):")
print(f"Raw: {np.mean(results['raw']):.4f} ± {np.std(results['raw']):.4f}")
print(f"Raw + PCA: {np.mean(results['pca']):.4f} ± {np.std(results['pca']):.4f}")
print(f"Features: {np.mean(results['features']):.4f} ± {np.std(results['features']):.4f}")
print(f"Features + PCA: {np.mean(results['features_pca']):.4f} ± {np.std(results['features_pca']):.4f}")