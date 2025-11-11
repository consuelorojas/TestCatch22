import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
plt.style.use('report.mplstyle')


result_file = "results/sine/sine_periods/results_20251103_155528.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)


# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["periods"]
    for method in ["raw", "pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"periods": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

#### error bars

markers = {
    "raw": "o", 
    "pca": "s", 
    "features": "D", 
    "features_pca": "^"
}

# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "periods"])
    .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"))
    .reset_index()
)


# Ensure numeric
df_grouped["periods"] = pd.to_numeric(df_grouped["periods"], errors="coerce")

plt.figure(figsize=(6.4, 4.8))

for method, marker in markers.items():
    data = df_grouped[df_grouped["Method"] == method].sort_values("periods")
    plt.errorbar(
        data["periods"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt=marker,
        capsize=5,
        #elinewidth=1,
        alpha=0.7,
        label=method
    )

plt.xlabel(r"Number of periods $(N_p)$")
plt.ylabel("AUC")
#plt.legend(ncol=2, loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.text(0.8, 1.0, "(b)", fontweight="bold", fontsize=14, va="bottom", ha="left")
plt.ylim(0.2, 1.1)
plt.savefig(
    "results/sine/sine_periods/errorbars_1-8_np.png", format="png",
    dpi=180
)
plt.show()
