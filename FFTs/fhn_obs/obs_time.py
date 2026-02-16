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



## sweep configuration
b0 = 0.1

b1 = 1.0
db1 = 0.032
#db1 = 0.178
b12 = b1 + db1
dt = 0.1


# step to subsampling
pseudo_period = 30
npp = 13          # change the number of points per period here!
step = int(pseudo_period / npp / dt)

epsilon = 0.2
I = 0
noise = 0.1

trans = 100 # transient

samples = 100


X, y, t = create_labeled_dataset([ #type: ignore
    (0, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
    (1, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, noise]})],
    n_samples_per_class=samples, subsample_step = step, transient = trans, return_time=True
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