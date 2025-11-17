import os
import sys
import pickle
from datetime import datetime
import tqdm as tqdm
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits




## sweep configuration
fbase = 5
dfreq = np.linspace(0.5, 10.0, 30)
f1 = fbase + dfreq

npoints = 7
nperiods = 3

noise = 0.1
samples = 100


# Run sweep
all_results = []
for i, freq in enumerate(tqdm.tqdm(f1,desc="Sweeping frequency differences")):
    X, y = create_labeled_dataset(
        [(0, 'sine', {'args': [fbase, 0.1, npoints, nperiods]}),
         (1, 'sine', {'args': [freq, 0.1, npoints, nperiods]})],
        n_samples_per_class=samples
    )
    splits = get_kfold_splits(X, y, n_splits=1, stratified=True)
    results = run_experiment(X, y, splits)

    all_results.append({
        'df': round(freq - fbase, 3),
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })


# plot results
df_records = []
for entry in all_results:
    delta_f = entry['df']
    for method in ['raw', 'pca', 'features', 'features_pca']:
        for auc in entry[method]:
            df_records.append({'Delta_Frequency': delta_f, 'Method': method, 'AUC': auc})  

df_results = pd.DataFrame(df_records)
methods = ['raw', 'pca', 'features', 'features_pca']

method_colors = {
    "raw": "C0", 
    "pca": "C1", 
    "features": "C2", 
    "features_pca": "C3"
}

method_labels = {
    "raw": "Raw",
    "pca": "PCA",
    "features": "Catch22",
    "features_pca": "Catch22 + PCA"
}

# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "Delta_Frequency"])
    .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"))
    .reset_index()
)

plt.figure(figsize=(6.4, 4.8))

#for method, color in method_colors.items():
method = 'pca'
data = df_grouped[df_grouped["Method"] == method]
plt.errorbar(
    data["Delta_Frequency"], data["AUC_mean"],
    yerr=data["AUC_std"],
    fmt='o',       # marker style
    capsize=5,          # error bar caps
    #elinewidth=1,       # error bar line thickness
    alpha=0.7,
    label=method_labels[method],
    color=method_colors[method]
)

plt.xlabel(r"Parameter Difference $(\nu - \nu_1)$")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
plt.ylim(0.2, 1.1)
plt.text(0.0, 1.0, "(a)", fontweight="bold", fontsize=14, va="bottom", ha="left")
plt.xticks(dfreq[::4].round(2))
#plt.xlim(-0.05, 0.65)
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/notebooks/sine_nu05_pca.png",
    format="png", dpi=180
)
legend = plt.legend(fontsize=14, loc="lower right")
#export_legend(legend)
plt.show()