import os
import sys
import time
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# -----------------------------
# PATHS
# -----------------------------
sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./preprocessing"))
sys.path.append(os.path.abspath("./features"))

from classification import time_single_fold
from dataset import create_labeled_dataset, get_kfold_splits


# -----------------------------
# FEATURES
# -----------------------------
def get_basic_stats(df):
    labels = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy()

    return pd.DataFrame({
        "mean": np.mean(X, axis=1),
        "variance": np.var(X, axis=1),
        "skew": skew(X, axis=1),
        "kurtosis": kurtosis(X, axis=1),
        "label": labels
    })


# -----------------------------
# SINGLE EXPERIMENT
# -----------------------------
def time_single_experiment(X, y):

    # -----------------------------
    # TYPE SAFETY
    # -----------------------------
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y))

    # CRITICAL: reset index to avoid KeyErrors
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    splits = get_kfold_splits(X, y, n_splits=10, stratified=True)

    train_times, test_times, tuning_times, preprocessing_times = [], [], [], []

    for train_idx, test_idx in splits:

        # -----------------------------
        # SAFE INDEXING (FIXES YOUR ERROR)
        # -----------------------------
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # -----------------------------
        # PREPROCESSING (feature extraction + scaling)
        # -----------------------------
        start = time.perf_counter()

        train_df = X_train.copy()
        train_df["label"] = y_train.values

        test_df = X_test.copy()
        test_df["label"] = y_test.values

        train_feat = get_basic_stats(train_df)
        test_feat = get_basic_stats(test_df)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_feat.drop(columns=["label"]))
        X_test_scaled = scaler.transform(test_feat.drop(columns=["label"]))

        preprocessing_times.append(time.perf_counter() - start)

        # -----------------------------
        # CLASSIFICATION
        # -----------------------------
        clf = SVC(probability=True)

        t_train, t_test, t_tune = time_single_fold(
            X_train_scaled,
            X_test_scaled,
            y_train.to_numpy(),
            y_test.to_numpy(),
            clf
        )

        train_times.append(t_train)
        test_times.append(t_test)
        tuning_times.append(t_tune)

    return list(zip(
        train_times,
        test_times,
        preprocessing_times,
        tuning_times
    ))


# -----------------------------
# PARALLEL EXECUTION
# -----------------------------
def run_all_experiments(signals_dict):

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(time_single_experiment)(X, y)
        for (X, y) in signals_dict.values()
    )

    return {
        name: res
        for name, res in zip(signals_dict.keys(), results)
    }


train = lambda arr: np.mean(arr) * 1000
std = lambda arr: np.std(arr) * 1000


def print_results(results):

    for signal in ["sine", "fhn_dyn", "fhn_obs"]:

        print(f"\n================ {signal.upper()} ================\n")

        print("Preprocessing: {:.2f} ± {:.2f} ms".format(
            train([x[2] for x in results[signal]]),
            std([x[2] for x in results[signal]])
        ))

        print("Train: {:.2f} ± {:.2f} ms".format(
            train([x[0] for x in results[signal]]),
            std([x[0] for x in results[signal]])
        ))

        print("Test:  {:.2f} ± {:.2f} ms".format(
            train([x[1] for x in results[signal]]),
            std([x[1] for x in results[signal]])
        ))

        print("Tuning: {:.2f} ± {:.2f} ms".format(
            train([x[3] for x in results[signal]]),
            std([x[3] for x in results[signal]])
        ))

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    samples = 100

    fbase = 5
    f1 = 5.18
    nperiods = 3
    npoints = 7

    b_base = 1
    b_dyn = b_base + 0.18
    b_obs = b_base + 0.03157

    epsilon = 0.2
    noise = 0.1

    step = 10
    trans = 100

    signals_dict = {}

    # ---------------- SINE ----------------
    X_sine, y_sine = create_labeled_dataset(
        [
            (0, 'sine', {'args': [fbase, 0.1, npoints, nperiods]}),
            (1, 'sine', {'args': [f1, 0.1, npoints, nperiods]})
        ],
        n_samples_per_class=samples
    )

    signals_dict["sine"] = (X_sine, y_sine)

    # ---------------- FHN DYN ----------------
    X_dyn, y_dyn, _ = create_labeled_dataset(
        [
            (0, 'fhn', {
                'length': 850,
                'dt': 0.1,
                'x0': [0, 0],
                'args': [b_base, b_base + 0.0, epsilon, 0, noise]
            }),
            (1, 'fhn', {
                'length': 850,
                'dt': 0.1,
                'x0': [0, 0],
                'args': [b_base, b_dyn, epsilon, 0, noise]
            })
        ],
        n_samples_per_class=samples,
        subsample_step=step,
        transient=trans,
        return_time=True
    )

    signals_dict["fhn_dyn"] = (X_dyn, y_dyn)

    # ---------------- FHN OBS ----------------
    X_obs, y_obs, _ = create_labeled_dataset(
        [
            (0, 'fhn_obs', {
                'length': 850,
                'dt': 0.1,
                'x0': [0, 0],
                'args': [b_base, b_base, epsilon, 0, noise]
            }),
            (1, 'fhn_obs', {
                'length': 850,
                'dt': 0.1,
                'x0': [0, 0],
                'args': [b_base, b_obs, epsilon, 0, noise]
            })
        ],
        n_samples_per_class=samples,
        subsample_step=step,
        transient=trans,
        return_time=True
    )

    signals_dict["fhn_obs"] = (X_obs, y_obs)

    # ---------------- RUN ----------------
    results = run_all_experiments(signals_dict)

    print_results(results)