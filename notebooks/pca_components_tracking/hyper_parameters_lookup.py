import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed


plt.style.use('report.mplstyle')
sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits


fbase = 5
deltaf = 0.01
f1 = [fbase + deltaf*i for i in range(0, 52, 2)]
dfreq = [deltaf *  i for i in range(0, 52, 2)]

npoints = 7
nperiods = 3

noise = 0.1
samples = 100

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed


# =========================================================
# UTIL: MODE
# =========================================================
def mode(series):
    return series.value_counts().idxmax()


# =========================================================
# SINGLE FOLD EVALUATION
# =========================================================
def evaluate_single_fold(X_train, X_test, y_train, y_test, n_components):

    # -----------------------
    # scaling
    # -----------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------
    # PCA
    # -----------------------
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # -----------------------
    # grid search
    # -----------------------
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
        SVC(kernel="rbf", probability=True),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_inner,
        n_jobs=1,
        refit=True
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    y_score = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_score)

    return {
        "auc": auc,
        "C": grid.best_params_["C"],
        "gamma": grid.best_params_["gamma"]
    }


# =========================================================
# ONE EXPERIMENT (ONE FREQUENCY)
# =========================================================
def run_single_experiment(freq):

    X, y = create_labeled_dataset(
        [
            (0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
            (1, 'sine', {'args': [freq, noise, npoints, nperiods]})
        ],
        n_samples_per_class=samples
    )

    splits = get_kfold_splits(
        X, y,
        n_splits=10,
        stratified=True
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(splits):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -----------------------
        # PCA = 2
        # -----------------------
        res = evaluate_single_fold(
            X_train, X_test, y_train, y_test,
            n_components=2
        )

        res.update({
            "freq": freq,
            "fold": fold,
            "pca": "2"
        })
        results.append(res)

        # -----------------------
        # PCA = 95%
        # -----------------------
        res = evaluate_single_fold(
            X_train, X_test, y_train, y_test,
            n_components=0.95
        )

        res.update({
            "freq": freq,
            "fold": fold,
            "pca": "95"
        })
        results.append(res)

    return pd.DataFrame(results)


# =========================================================
# PARALLEL SWEEP
# =========================================================
def run_sweep(frequency_sweep):

    dfs = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_single_experiment)(freq)
        for freq in frequency_sweep
    )

    return pd.concat(dfs, ignore_index=True)


# =========================================================
# AGGREGATION (MODE + STABILITY)
# =========================================================
def aggregate_results(df):

    agg = df.groupby(["freq", "pca"]).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),

        C_mode=("C", mode),
        gamma_mode=("gamma", mode),

        stability_C=("C", lambda x: x.value_counts(normalize=True).iloc[0]),
        stability_gamma=("gamma", lambda x: x.value_counts(normalize=True).iloc[0]),

        n_folds=("auc", "count")
    ).reset_index()

    return agg


# =========================================================
# PLOTTING
# =========================================================
def plot_aggregated(df, color_by="C_mode", stability_col="stability_C"):

    markers = {
        "2": "o",
        "95": "^"
    }

    unique_vals = sorted(df[color_by].unique(), key=str)

    cmap = plt.cm.viridis
    colors = {
        v: cmap(i / max(1, len(unique_vals) - 1))
        for i, v in enumerate(unique_vals)
    }

    plt.figure(figsize=(9, 5))

    for _, row in df.iterrows():

        plt.scatter(
            row["freq"],
            row["auc_mean"],
            marker=markers[row["pca"]],
            color=colors[row[color_by]],
            alpha=row[stability_col],
            s=110,
            edgecolor="black",
            linewidth=0.3
        )

    # legend: hyperparameter
    handles = [
        Line2D([], [], marker="o", linestyle="",
               color=colors[v], label=f"{color_by}={v}")
        for v in unique_vals
    ]

    # legend: PCA
    handles += [
        Line2D([], [], marker="o", linestyle="", color="black", label="PCA=2"),
        Line2D([], [], marker="^", linestyle="", color="black", label="PCA=95%")
    ]

    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xlabel("Frequency")
    plt.ylabel("Mean AUC")
    plt.title("Performance + Hyperparameter Stability (mode + transparency)")

    plt.tight_layout()
    plt.savefig(f"hyper_paramater_{color_by}.png", dpi=300)
    plt.show()


# =========================================================
# RUN EVERYTHING
# =========================================================
results_df = run_sweep(f1)

agg_df = aggregate_results(results_df)

print(agg_df.head())

plot_aggregated(agg_df, color_by="C_mode", stability_col="stability_C")
plot_aggregated(agg_df, color_by="gamma_mode", stability_col="stability_gamma")