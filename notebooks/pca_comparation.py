import os
import sys
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('report.mplstyle')

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./preprocessing"))
sys.path.append(os.path.abspath("./features"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits
from preprocessing import apply_pca
from features import extract_features


######### SINE WAVES  #########
## sweep configuration
fbase = 5
f1 = 5.5
nperiods = 3
npoints = 7

noise = 0.1
samples = 100


# Run sweep
X, y = create_labeled_dataset(
    [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
    (1, 'sine', {'args': [f1, noise, npoints, nperiods]})],
    n_samples_per_class=samples
)

splits = get_kfold_splits(X, y, n_splits=1, stratified=True)

x_train, x_test = X[splits[0][0]], X[splits[0][1]]
y_train, y_test = y[splits[0][0]], y[splits[0][1]]

# Raw + PCA
train_pca, pca_tf, scaler = apply_pca(x_train,n_components = 2)
test_pca = scaler.transform(x_test)
test_pca = pca_tf.transform(test_pca)


# Features + PCA
X_feat = extract_features(X, return_array=True)
train_feat, test_feat = X_feat[splits[0][0]], X_feat[splits[0][1]]
train_feat_pca, pca_tf, scaler = apply_pca(train_feat, n_components=2)
test_feat = test_feat.values if isinstance(test_feat, pd.DataFrame) else test_feat
test_feat_pca = scaler.transform(test_feat)
test_feat_pca = pca_tf.transform(test_feat_pca)

# Plotting
fig, axs = plt.subplots(1, 2)#, figsize=(10, 8))
axs[0].scatter(train_pca[:, 0], train_pca[:, 1], c=y_train, cmap='coolwarm', label='Train PCA')
axs[0].scatter(test_pca[:, 0], test_pca[:, 1], c=y_test, cmap='viridis', marker='x', label='Test PCA')
axs[0].set_title('Raw Data + PCA')
axs[0].legend()

axs[1].scatter(train_feat_pca[:, 0], train_feat_pca[:, 1], c=y_train, cmap='coolwarm', label='Train Features PCA')
axs[1].scatter(test_feat_pca[:, 0], test_feat_pca[:, 1], c=y_test, cmap='viridis', marker='x', label='Test Features PCA')
axs[1].set_title('Features + PCA')
axs[1].legend()

plt.tight_layout()
plt.show()