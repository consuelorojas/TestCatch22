import os
import sys
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import time_experiment
from dataset import create_labeled_dataset, get_kfold_splits

fbase = 1
df = 0.18
f1 = fbase + df

npoints = 10
nperiods = 3

noise = 0.1

samples = 100



X, y = create_labeled_dataset( # type: ignore
    [(0, 'sine', {'args': [fbase, 0.1, npoints, nperiods]}),
        (1, 'sine', {'args': [f1, 0.1, npoints, nperiods]})],
    n_samples_per_class=samples
)


splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
preprocessing_time = []
train_time = []
test_time = []
for train_idx, test_idx in splits:
    time_0 = time.time()
    X_fft = np.abs(np.fft.rfft(X, axis=1))

    x_train, x_test = X_fft[train_idx], X_fft[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    preprocessing_time.append(time.time() - time_0)

    clf = SVC(probability=True, random_state=42)
    time_1 = time.time()
    clf.fit(x_train, y_train)
    train_time.append(time.time() - time_1)

    time_2 = time.time()
    y_pred = clf.predict_proba(x_test)[:, 1]
    test_time.append(time.time() - time_2)

print(f"Average preprocessing time: {np.mean(preprocessing_time) * 1000:.4f} ms")
print(f"Standard deviation of preprocessing time: {np.std(preprocessing_time) * 1000:.4f} ms")
print(f"Average training time: {np.mean(train_time) * 1000:.4f} ms")
print(f"Standard deviation of training time: {np.std(train_time) * 1000:.4f} ms")
print(f"Average test time: {np.mean(test_time) * 1000:.4f} ms")
print(f"Standard deviation of test time: {np.std(test_time) * 1000:.4f} ms")