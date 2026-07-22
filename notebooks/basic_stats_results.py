import os
import sys
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath("./data"))

from dataset import create_labeled_dataset, get_kfold_splits



# =============================================================================
# Sweep configuration
# =============================================================================

b0 = 0.1

b1 = 1
#b12 = np.linspace(b1, 1.2, 20) #observational noise


db1 = 0.2
#dynamic noise
b12s = np.linspace(b1, 1.2, 10)
#b12 = np.concatenate((b12s, np.array([1.25, 1.3, 1.35, 1.4])))

epsilon = 0.2
I = 0
noise = 0.1

dt = 0.1

pseudo_period = 30
npp = 10
step = int(pseudo_period / npp / dt)

trans = 50
samples = 100


#Sine
fbase = 5
deltaf = 0.01
b12 = [fbase + deltaf*i for i in range(0, 52, 2)] #frequecnias
dfreq = [deltaf *  i for i in range(0, 52, 2)]

npoints = 7
nperiods = 3

# =============================================================================
# Output directory
# =============================================================================

sweep_name = "sine/basic_stats"

output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(
    output_dir,
    f"results_{timestamp}.pkl"
)


# =============================================================================
# Feature extraction
# =============================================================================

def extract_basic_features(X):

    return np.column_stack([
        np.mean(X, axis=1),
        np.var(X, axis=1),
        skew(X, axis=1),
        kurtosis(X, axis=1),
    ])


# =============================================================================
# Single experiment
# =============================================================================

def run_single_experiment(b):

    '''
    X, y = create_labeled_dataset(
        [
            (
                0,
                "fhn",
                {
                    "length": 750,
                    "dt": 0.1,
                    "x0": [0, 0],
                    "args": [b0, b1, epsilon, I, noise],
                },
            ),
            (
                1,
                "fhn",
                {
                    "length": 750,
                    "dt": 0.1,
                    "x0": [0, 0],
                    "args": [b0, b, epsilon, I, noise],
                },
            ),
        ],
        n_samples_per_class=samples,
        subsample_step=step,
        transient=trans,
    )
    '''

    X, y = create_labeled_dataset( #type:ignore
    [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
        (1, 'sine', {'args': [b, noise, npoints, nperiods]})],
    n_samples_per_class=samples
    )
    
    splits = get_kfold_splits(
        X,
        y,
        n_splits=10,
        stratified=True,
    )

    aucs = []

    for train_idx, test_idx in splits:

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # -----------------------------------
        # Statistical features
        # -----------------------------------

        X_train = extract_basic_features(X_train)
        X_test = extract_basic_features(X_test)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        param_grid = {
            "C": [0.1, 1, 10],
            "gamma": ["scale", 0.1, 1],
        }

        grid = GridSearchCV(
            estimator=SVC(
                kernel="rbf",
                probability=True,
            ),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=1,          # IMPORTANT: avoid nested parallelism
        )

        grid.fit(X_train, y_train)

        scores = grid.best_estimator_.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, scores)

        aucs.append(auc)

    return {
        "b": round(b - b1, 3),
        "auc": aucs,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    all_results = []

    with ProcessPoolExecutor() as executor:

        futures = {
            executor.submit(run_single_experiment, b): b
            for b in b12
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Sweeping parameter b",
        ):
            all_results.append(future.result())

    all_results = sorted(all_results, key=lambda x: x["b"])

    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Results saved to:\n{output_file}")

    # =============================================================================
    # Plot results
    # =============================================================================

    import pandas as pd
    import matplotlib.pyplot as plt

    plt.style.use("report.mplstyle")

    records = []

    for entry in all_results:
        for auc in entry["auc"]:
            records.append(
                {
                    "b": entry["b"],
                    "AUC": auc,
                }
            )

    df = pd.DataFrame(records)

    summary = (
        df.groupby("b")
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.errorbar(
        summary["b"],
        summary["AUC_mean"],
        yerr=summary["AUC_std"],
        fmt="o",
        color="C6",
        capsize=5,
        alpha=0.8,
        #label="Statistical features",
    )

    ax.set_xlabel(r"Parameter difference $(\nu-\nu_0)$")
    ax.set_ylabel("AUC")

    ax.set_ylim(0.2, 1.05)
    ax.set_xlim(summary["b"].min() - 0.005, summary["b"].max() + 0.005)

    ax.grid(True)

    #ax.legend()

    plt.tight_layout()

    figure_file = output_file.replace(".pkl", ".eps")

    plt.savefig(
        figure_file,
        format="eps",
        dpi=300,
    )
    plt.show()
    plt.close(fig)

    print(f"Figure saved to:\n{figure_file}")


if __name__ == "__main__":
    main()