import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/sine/sine_noise/results_20251112_112335.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["noise"]
    for method in ["raw", "pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"noise": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "noise"])
    .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"))
    .reset_index()
)

# --- Plot with error bars ---
markers = {
    "raw": "o", 
    "pca": "s", 
    "features": "D", 
    "features_pca": "^"
}

method_colors = {
    "raw": "C0", 
    "pca": "C1", 
    "features": "C2", 
    "features_pca": "C3"
}

plt.figure(figsize=(6.4, 4.8))

for method, color in method_colors.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["noise"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt='o',         # marker style
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        color=color
    )

plt.xlabel(r"Noise strength $(D)$")
plt.ylabel("AUC")
#plt.legend(loc = 'lower left')
plt.grid(True)
plt.tight_layout()
plt.ylim(0.2, 1.1)
plt.xlim(-0.01, 0.42)
plt.text(0.0, 1.0, "(d)", fontweight="bold", fontsize=14, va="bottom", ha="left")
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/sine/sine_noise/errorbars_noise_resolution.eps",
    format="eps", dpi=180
)
plt.show()
