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

print("Timing Results (milliseconds):")

train = lambda arr: np.mean(arr) * 1000
std = lambda arr: np.std(arr) * 1000

print("Raw train: {:.2f} ± {:.2f} ms".format(train([x[0] for x in results['raw']]), std([x[0] for x in results['raw']])))
print("Raw test:  {:.2f} ± {:.2f} ms".format(train([x[1] for x in results['raw']]), std([x[1] for x in results['raw']])))

print("Raw + PCA train: {:.2f} ± {:.2f} ms".format(train([x[0] for x in results['pca']]), std([x[0] for x in results['pca']])))
print("Raw + PCA test:  {:.2f} ± {:.2f} ms".format(train([x[1] for x in results['pca']]), std([x[1] for x in results['pca']])))

print("Features train: {:.2f} ± {:.2f} ms".format(train([x[0] for x in results['features']]), std([x[0] for x in results['features']])))
print("Features test:  {:.2f} ± {:.2f} ms".format(train([x[1] for x in results['features']]), std([x[1] for x in results['features']])))

print("Features + PCA train: {:.2f} ± {:.2f} ms".format(train([x[0] for x in results['features_pca']]), std([x[0] for x in results['features_pca']])))
print("Features + PCA test:  {:.2f} ± {:.2f} ms".format(train([x[1] for x in results['features_pca']]), std([x[1] for x in results['features_pca']])))
