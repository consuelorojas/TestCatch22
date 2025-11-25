import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

# ---- Load results from file ----

sine = "notebooks/times/fhn_samples/sine_samples_times_results.pkl"
fhn_obs = "notebooks/times/fhn_samples/fhn_obs_samples_times_results.pkl"
fhn_dyn = "notebooks/times/fhn_samples/fhn_dyn_samples_times_results.pkl"

def extract_results(results):
    with open(results, "rb") as f:
        results = pickle.load(f)
    records = []
    for entry in results:
        sample_size = entry["samples"]
        for method in ["raw", "pca", "features", "features_pca"]:
            for auc in entry[method]:
                records.append({"samples": sample_size, "Method": method, "AUC": auc})
    return pd.DataFrame(records)

df_sine = extract_results(sine)
df_fhn_obs = extract_results(fhn_obs)
df_fhn_dyn = extract_results(fhn_dyn)

