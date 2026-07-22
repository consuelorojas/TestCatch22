from pathlib import Path

"""
linear_svm_feature_weights.py

Computes feature importance from the internal weights of a Linear SVM.

The script:
- creates Sine, FHN observational noise, and FHN dynamic noise datasets
- extracts features
- trains Linear SVM models using repeated folds
- averages absolute coefficients across folds
- generates a normalized heatmap

The coefficients are interpretable because the model is linear:
importance(feature_i) = |w_i|
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

plt.style.use("report.mplstyle")

# project modules
sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./features"))
sys.path.append(os.path.abspath("./preprocessing"))

from dataset import create_labeled_dataset, get_kfold_splits
from features import extract_features

cmap = mpl.colormaps.get_cmap("coolwarm").with_extremes(under="w")
cmap.set_bad("0.3")

# ============================================================
# Parameters
# ============================================================

samples = 100
noise = 0.1

# sine
fbase = 5
f1 = 5.18
nperiods = 3
npoints = 7

# FHN
b0 = 1
b_obs = 1.032
b_dyn = 1.175

epsilon = 0.2
I = 0

dt = 0.1
trans = 100

pseudo_period = 30
npp = 10
step = int(pseudo_period / npp / dt)


# ============================================================
# Dataset generation
# ============================================================

Xs, ys = create_labeled_dataset(
    [
        (0, "sine",
         {"args": [fbase, noise, npoints, nperiods]}),
        (1, "sine",
         {"args": [f1, noise, npoints, nperiods]}),
    ],
    n_samples_per_class=samples
)


Xobs, yobs = create_labeled_dataset(
    [
        (0, "fhn_obs",
         {"length":850, "dt":dt, "x0":[0,0],
          "args":[0.1, b0, epsilon, I, noise]}),
        (1, "fhn_obs",
         {"length":850, "dt":dt, "x0":[0,0],
          "args":[0.1, b_obs, epsilon, I, noise]}),
    ],
    n_samples_per_class=samples,
    subsample_step=step,
    transient=trans
)


Xdyn, ydyn = create_labeled_dataset(
    [
        (0, "fhn_obs",
         {"length":850, "dt":dt, "x0":[0,0],
          "args":[0.1, b0, epsilon, I, noise]}),
        (1, "fhn_obs",
         {"length":850, "dt":dt, "x0":[0,0],
          "args":[0.1, b_dyn, epsilon, I, noise]}),
    ],
    n_samples_per_class=samples,
    subsample_step=step,
    transient=trans
)


# ============================================================
# Feature extraction
# ============================================================

Sfeat = extract_features(Xs)
Obsfeat = extract_features(Xobs)
Dynfeat = extract_features(Xdyn)


# ============================================================
# Linear SVM weights
# ============================================================

def linear_svm_weights(X, y):

    fold_weights = []

    splits = get_kfold_splits(
        X.values,
        y,
        n_splits=10,
        stratified=True
    )

    for train_idx, test_idx in splits:

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        C=1,
                        dual=False,
                        random_state=13
                    )
                )
            ]
        )

        model.fit(
            X_train,
            y_train
        )

        coef = np.abs(
            model.named_steps["svm"].coef_[0]
        )

        fold_weights.append(coef)


    return pd.Series(
        np.mean(fold_weights, axis=0),
        index=X.columns
    )


# ============================================================
# Compute weights
# ============================================================

S_weights = linear_svm_weights(Sfeat, ys)
Obs_weights = linear_svm_weights(Obsfeat, yobs)
Dyn_weights = linear_svm_weights(Dynfeat, ydyn)


weights = pd.DataFrame(
    {
        "Sine": S_weights,
        "FHN Obs": Obs_weights,
        "FHN Dyn": Dyn_weights,
    }
)


# ============================================================
# Normalize and plot
# ============================================================

weights_norm = weights.div(
    weights.max(axis=0),
    axis=1
)

'''weights_norm = weights_norm.loc[
    weights_norm.mean(axis=1)
    .sort_values(ascending=False)
    .index
]'''

order = ['acf_timescale', 'centroid_freq', 'low_freq_power', 'forecast_error',
       'acf_first_min', 'transition_matrix', 'ami2', 'stretch_decreasing',
       'high_fluctuation', 'stretch_high', 'mode_5', 'trev',
       'outlier_timing_neg', 'embedding_dist', 'outlier_timing_pos', 'mode_10',
       'entropy_pairs', 'periodicity', 'ami_timescale', 'whiten_timescale',
       'rs_range', 'dfa']

weights_order = weights.loc[order]

plt.figure(figsize=(8,10))

sns.heatmap(
    weights_order,
    cmap=cmap,
    #vmin=0,
    #vmax=1,
    square=False,
    linewidths=1.0,
    linecolor="white",
    cbar_kws={
        "label": "Linear SVM weight"
    }
)


plt.xlabel("")
plt.ylabel("")

plt.tight_layout()
plt.grid(False)

plt.savefig(
    "linear_svm_feature_weights_heatmap.eps",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


weights_norm.to_csv(
    "linear_svm_feature_weights.csv"
)

print("Finished.")


