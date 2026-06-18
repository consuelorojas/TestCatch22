import numpy as np
import pandas as pd
import pickle
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


# results 

PARAMETERS = ["b", "df", "noise", "npp", "periods", "samples", "npoints"]

METHODS = [
    "raw",
    "pca",
    "fft",
    "fft_pca",
    "features",
    "features_pca"
]

from pathlib import Path

def latest_pkl(folder):
    folder = Path(folder)
    files = list(folder.glob("results_*.pkl"))

    if not files:
        raise FileNotFoundError(f"No .pkl files found in {folder}")

    return str(max(files, key=lambda p: p.stat().st_mtime))

# open results
def results_dataframe(path_to_results):

    with open(path_to_results, "rb") as f:
        all_results = pickle.load(f)
    # Detect which parameter is present
    first_entry = all_results[0]

    param_col = None

    for p in PARAMETERS:
        if p in first_entry:
            param_col = p
            break

    if param_col is None:
        raise ValueError(
            f"Could not find any parameter column in {PARAMETERS}"
        )

    records = []
    for entry in all_results:
        param_value = entry[param_col]
        for method in METHODS:
            for run_id, auc in enumerate(entry[method]):
                records.append({
                    param_col: param_value,
                    "run": run_id,
                    "Method": method,
                    "AUC": auc
                })

    return pd.DataFrame(records), param_col


# remove trivial regions
def remove_ceiling_region(df_results, param_col, ceiling_auc=0.99):

    means = (
        df_results
        .groupby([param_col, "Method"])["AUC"]
        .mean()
        .unstack()
    )

    valid_params = means.index[
        ~(means >= ceiling_auc).all(axis=1)
    ]

    return df_results[
        df_results[param_col].isin(valid_params)
    ].copy()




def compute_statistic(df, param_col, remove_ceiling=True, ceiling_auc=0.99):

    if remove_ceiling:
        df = remove_ceiling_region(df, param_col, ceiling_auc)


    pivot = (
        df
        .pivot_table(
            index=[param_col, "run"],
            columns='Method',
            values = 'AUC'
        )
        .dropna()
    )

    methods = pivot.columns.tolist()

    # friedman test
    friedman_test, friedman_p = friedmanchisquare(*[pivot[m] for m in methods])

    # wilcoxon pairwise
    results = []

    for m1, m2 in combinations(methods,2):
        _, p_raw = wilcoxon(pivot[m1], pivot[m2])
        results.append(
            {
                "Method_1": m1,
                "Method_2": m2,
                "p_raw": p_raw
            }
        )
        
    df_posthoc = pd.DataFrame(results)

    reject, p_corr, _, _ = multipletests(
        df_posthoc["p_raw"],
        method='holm'
    )

    df_posthoc['p_corrected'] = p_corr
    df_posthoc['significant'] = reject


    return {
        "parameter": param_col,
        "n_blocks": len(pivot),
        "friedman_stat": float(friedman_test),
        "friedman_p": float(friedman_p),
        "posthoc": df_posthoc.sort_values(
            "p_corrected"
        ).reset_index(drop=True)
    }


# single experiment

def process_experiment(path_to_results, remove_ceiling=True, ceiling_auc=0.99):
    
    df_results, param_col = results_dataframe(path_to_results)

    stats = compute_statistic(df_results, param_col, remove_ceiling=remove_ceiling, ceiling_auc=ceiling_auc)

    return stats

# process the dictionary

def process_results_dict(results_dict, max_workers = 4, remove_ceiling=True, ceiling_auc=0.99, save_path=None):

    all_stats = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(
                process_experiment,
                path,
                remove_ceiling,
                ceiling_auc
            ): experiment_name
            for experiment_name, path in results_dict.items()
        }

        for future in as_completed(futures):
            experiment_name = futures[future]
            try:
                all_stats[
                    experiment_name
                ] = future.result()

            except Exception as e:
                all_stats[
                    experiment_name
                ] = {"error": str(e)}
            
        if save_path is not None:

            with open(save_path,"wb") as f:
                pickle.dump(all_stats,f)
        
        return all_stats


def main():

    results_sine = {
        'df': latest_pkl('results/sine/sine_frequency'),
        'periods': latest_pkl('results/sine/sine_periods'),
        'noise': latest_pkl('results/sine/sine_noise'),
        'npp': latest_pkl('results/sine/sine_points'),
        'samples': latest_pkl('results/sine/sine_samples')
    }

    results_fhn_dyn = {
        'b': latest_pkl('results/fhn/fhn_parameter'),
        'periods': latest_pkl('results/fhn/fhn_periods'),
        'npp': latest_pkl('results/fhn/fhn_npp'),
        'noise': latest_pkl('results/fhn/fhn_noise'),
        'samples': latest_pkl('results/fhn/fhn_samples')
    }

    results_fhn_obs = {
        'b': latest_pkl('results/fhn_obs/parameter'),
        'noise': latest_pkl('results/fhn_obs/noise'),
        'npp': latest_pkl('results/fhn_obs/npoints'),
        'periods': latest_pkl('results/fhn_obs/periods'),
        'samples': latest_pkl('results/fhn_obs/samples')
    }



    all_statistics = {
        "fhn_dyn": process_results_dict(
            results_fhn_dyn,
            max_workers=4,
            remove_ceiling=True,
            ceiling_auc=0.99,
            save_path='fhn_dyn_stats.pkl'
        ),
        "sine": process_results_dict(
            results_sine,
            max_workers=4,
            remove_ceiling=True,
            ceiling_auc=0.99,
            save_path='sine_stats.pkl'
        ),
        "fhn_obs": process_results_dict(
            results_fhn_obs,
            max_workers=4,
            remove_ceiling=True,
            ceiling_auc = 0.99,
            save_path='fhn_obs_stats.pkl'
        )
    }


    with open("all_statistics.pkl", "wb") as f:
        pickle.dump(all_statistics, f)

    print("✔ All analyses completed and saved.")




if __name__=="__main__":
    main()