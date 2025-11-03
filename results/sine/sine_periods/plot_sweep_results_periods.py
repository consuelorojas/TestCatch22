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
'''
# ---- Boxplot ----
plt.figure(figsize=(20, 14))
sns.boxplot(data=df_results, x="periods", y="AUC", hue="Method", palette="Set2")
plt.title("AUC across different periods by method")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, axis="y")
plt.legend(title="Method")
plt.ylim(-0.1, 1.1)
plt.show()

# ---- Scatter plot with markers ----
markers = {
    "raw": "o", 
    "pca": "s", 
    "features": "D", 
    "features_pca": "^"
}

plt.figure(figsize=(20, 14))

for method, marker in markers.items():
    data = df_results[df_results["Method"] == method]
    plt.scatter(
        data["periods"], data["AUC"],
        label=method,
        marker=marker,
        alpha=0.7
    )

#plt.title("AUC vs Frequency Difference (Δf)")
plt.xlabel(r"Number of periods $(N_p)$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
#plt.xlim(-0.05, 0.65)
#plt.savefig(f"results/sine/sine_periods/auc_vs_np_{datetime.now()}.png", dpi=180)
plt.show()

'''
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
    "results/sine/sine_periods/errorbars_1-8_np.eps", format="eps",
    dpi=180
)
plt.show()
