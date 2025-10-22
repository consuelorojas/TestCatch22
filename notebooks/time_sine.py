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