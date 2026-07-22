from pathlib import Path



import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

plt.style.use("report.mplstyle")

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./features"))
sys.path.append(os.path.abspath("./preprocessing"))

from dataset import create_labeled_dataset, get_kfold_splits
from features import extract_features

cmap = mpl.colormaps.get_cmap("coolwarm").with_extremes(under="w")
cmap.set_bad("0.3")

# ============================================================
# Configuration
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
b_dyn = 1.09

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
        (0, "sine", {"args": [fbase, noise, npoints, nperiods]}),
        (1, "sine", {"args": [f1, noise, npoints, nperiods]}),
    ],
    n_samples_per_class=samples,
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
    transient=trans,
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
    transient=trans,
)


# ============================================================
# Feature extraction
# ============================================================

Sfeat = extract_features(Xs)
Obsfeat = extract_features(Xobs)
Dynfeat = extract_features(Xdyn)


# ============================================================
# Cross validated permutation importance
# ============================================================

def svm_permutation_importance(X, y):

    results = []

    splits = get_kfold_splits(
        X.values,
        y,
        n_splits=10,
        stratified=True,
    )

    for train_idx, test_idx in splits:

        Xtrain = X.iloc[train_idx]
        Xtest = X.iloc[test_idx]

        ytrain = y[train_idx]
        ytest = y[test_idx]


        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        C=1,
                        gamma="scale",
                        random_state=13,
                    ),
                ),
            ]
        )

        model.fit(Xtrain, ytrain)


        imp = permutation_importance(
            model,
            Xtest,
            ytest,
            scoring="roc_auc",
            n_repeats=50,
            random_state=13,
            n_jobs=-1,
        )

        results.append(imp.importances_mean)


    return pd.Series(
        np.mean(results, axis=0),
        index=X.columns
    )


S_imp = svm_permutation_importance(Sfeat, ys)
Obs_imp = svm_permutation_importance(Obsfeat, yobs)
Dyn_imp = svm_permutation_importance(Dynfeat, ydyn)


# ============================================================
# Plot
# ============================================================

importance = pd.DataFrame(
    {
        "Sine": S_imp,
        "FHN Obs": Obs_imp,
        "FHN Dyn": Dyn_imp,
    }
)


# order by average importance
importance = importance.loc[
    importance.mean(axis=1).sort_values(ascending=False).index
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(12,6),
    sharey=True
)


for ax, col in zip(
    axes,
    ["Sine", "FHN Obs", "FHN Dyn"]
):

    sns.barplot(
        x=importance[col],
        y=importance.index,
        ax=ax,
    )

    ax.set_title(col)
    ax.set_xlabel("Permutation importance")
    ax.grid(False)


axes[0].set_ylabel("Feature")
axes[1].set_ylabel("")
axes[2].set_ylabel("")

plt.tight_layout()

plt.savefig(
    "svm_permutation_importance.eps",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ============================================================
# Normalize importance per classifier
# ============================================================

importance_norm = importance.copy()

importance_norm = importance_norm.div(
    importance_norm.max(axis=0),
    axis=1
)

# ============================================================
# Desired feature order
# ============================================================

order = [
    'acf_timescale', 'centroid_freq', 'low_freq_power', 'forecast_error',
    'acf_first_min', 'transition_matrix', 'ami2', 'stretch_decreasing',
    'high_fluctuation', 'stretch_high', 'mode_5', 'trev',
    'outlier_timing_neg', 'embedding_dist', 'outlier_timing_pos', 'mode_10',
    'entropy_pairs', 'periodicity', 'ami_timescale', 'whiten_timescale',
    'rs_range', 'dfa'
]

importance_norm = importance_norm.loc[order]
importance_order = importance.loc[order]

# ============================================================
# Heatmap
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 10)
)

sns.heatmap(
    importance_order,
    cmap=cmap,
    #vmin=0,
    #vmax=1,
    linewidths=0.2,
    cbar_kws={
        "label": "Permutation importance"
    },
    ax=ax,
)


ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.grid(False)

plt.savefig(
    "svm_permutation_importance_heatmap.eps",
    dpi=300,
    bbox_inches="tight",
)


plt.show()


# ============================================================
# Save table
# ============================================================

importance.to_csv(
    "svm_permutation_importance.csv"
)

print("Finished.")

