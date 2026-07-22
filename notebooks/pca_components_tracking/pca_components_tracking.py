import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed

plt.style.use("report.mplstyle")

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from dataset import create_labeled_dataset, get_kfold_splits

'''
##############################################################
# PARAMETERS
##############################################################


fbase = 5
deltaf = 0.01

f1 = [fbase + deltaf * i for i in range(0, 52, 2)]
df1 = [f - fbase for f in f1]

npoints = 7
nperiods = 3
noise = 0.1
samples = 100
'''

##############################################################
# FHN
##############################################################

b0 = 0.1

b1 = 1
db1 = 0.005
#b12 = np.linspace(b1, 1.2, 20) # observational
b12s = np.linspace(b1, 1.2, 10)
b12 = np.concatenate((b12s, np.array([1.25, 1.3, 1.35, 1.4])))

deltab12 = b12 - b1

epsilon = 0.2
I = 0
noise = 0.1

# simulation parameters
dt = 0.1

# step to subsampling
pseudo_period = 30
npp = 10            # change the number of points per period here
step = int(pseudo_period / npp / dt)

trans = 50
samples = 100


##############################################################
# SINGLE FOLD
##############################################################

def evaluate_single_fold(X_train, X_test, y_train, y_test):

    # -------------------------
    # Scale
    # -------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # PCA (95%)
    # -------------------------
    pca = PCA(n_components=0.99)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    n_components = pca.n_components_

    # -------------------------
    # Grid Search
    # -------------------------
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1]
    }

    cv_inner = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=SVC(kernel="rbf", probability=True),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_inner,
        n_jobs=1,
        refit=True
    )

    grid.fit(X_train, y_train)

    y_score = grid.best_estimator_.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_score)

    return {
        "auc": auc,
        "n_components": n_components
    }


##############################################################
# ONE FREQUENCY
##############################################################

def run_single_experiment(freq):

    X, y = create_labeled_dataset([
        (0, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
        (1, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, freq, epsilon, I, noise]})],
        n_samples_per_class=samples, subsample_step = step, transient = trans
        )
        
    '''X, y = create_labeled_dataset( #type:ignore
        [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
         (1, 'sine', {'args': [freq, noise, npoints, nperiods]})],
        n_samples_per_class=samples
    )
    '''

    splits = get_kfold_splits(
        X,
        y,
        n_splits=10,
        stratified=True
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(splits):

        res = evaluate_single_fold(
            X[train_idx],
            X[test_idx],
            y[train_idx],
            y[test_idx]
        )

        res["freq"] = freq-b1
        res["fold"] = fold

        results.append(res)

    return pd.DataFrame(results)


##############################################################
# PARALLEL SWEEP
##############################################################

def run_sweep(freqs):

    dfs = Parallel(n_jobs=10, backend="loky")(
        delayed(run_single_experiment)(freq)
        for freq in freqs
    )

    return pd.concat(dfs, ignore_index=True)


##############################################################
# AGGREGATION
##############################################################

def aggregate_results(df):

    agg = (
        df.groupby("freq")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            mean_components=("n_components", "mean"),
            std_components=("n_components", "std")
        )
        .reset_index()
    )

    return agg


##############################################################
# PLOT
##############################################################

def plot_results(df):

    fig, ax = plt.subplots(figsize=(8,5))

    ax.errorbar(
        df["freq"],
        df["auc_mean"],
        yerr=df["auc_std"],
        fmt="none",          # don't draw markers
        ecolor="black",
        elinewidth=1,
        capsize=3,
        alpha=0.6,
        zorder=1
    )

    scatter = ax.scatter(
        df["freq"],
        df["auc_mean"],
        c=df["mean_components"],
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidth=0.4,
        zorder = 2
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Mean PCA components (99% variance)")

    ax.set_xlabel(r"Parameter difference ($b - b_0$)")
    ax.set_ylabel("Mean AUC")

    ax.set_ylim(0.45, 1.02)

    plt.tight_layout()
    plt.savefig("fhn_dyn_auc_vs_frequency_pca_components99.eps", dpi=300)
    plt.show()


##############################################################
# RUN
##############################################################

results_df = run_sweep(b12)

print(results_df.head())

agg_df = aggregate_results(results_df)

print(agg_df)

plot_results(agg_df)