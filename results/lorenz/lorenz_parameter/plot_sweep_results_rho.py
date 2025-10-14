import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace with your Lorenz results path:
result_file = "results/lorenz/lorenz_parameter/results_20250920_020611.pkl"

with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    param_diff = entry["rho"]   # difference in rho relative to 28
    for method in ["raw", "pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"Δρ": param_diff, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

# ---- Boxplot ----
plt.figure(figsize=(20, 14))
sns.boxplot(data=df_results, x="Δρ", y="AUC", hue="Method", palette="Set2")
plt.title("AUC across Lorenz parameter differences (ρ - 28) by method")
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

plt.figure(figsize=(10, 6))

for method, marker in markers.items():
    data = df_results[df_results["Method"] == method]
    plt.scatter(
        data["Δρ"], data["AUC"],
        label=method,
        marker=marker,
        alpha=0.7
    )

plt.xlabel(r"Parameter Difference $(\rho - 28)$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
plt.savefig("results/lorenz/lorenz_parameter/auc_vs_rho_diff.png", dpi=180)
plt.show()

# --- Compute mean & std per method/Δρ ---
df_grouped = (
    df_results
    .groupby(["Method", "Δρ"])
    .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"))
    .reset_index()
)

# --- Plot with error bars ---
plt.figure(figsize=(10, 6))

for method, marker in markers.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["Δρ"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt=marker,
        capsize=4,
        elinewidth=1,
        alpha=0.7,
        label=method
    )

plt.xlabel(r"Parameter Difference $(\rho - 28)$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
plt.xlim(27.9, 29.1)
plt.savefig("results/lorenz/lorenz_parameter/auc_vs_rho_diff_errorbars.png", dpi=180)
plt.show()
