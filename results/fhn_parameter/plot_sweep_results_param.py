import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mlpstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/fhn_parameter/results_20250609_120140.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["b"]
    for method in ["raw", "pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"b": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

# ---- Boxplot ----
plt.figure(figsize=(20, 14))
sns.boxplot(data=df_results, x="b", y="AUC", hue="Method", palette="Set2")
#plt.title("AUC across frequency differences by method")
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
        data["b"], data["AUC"],
        label=method,
        marker=marker,
        alpha=0.7
    )

#plt.title("AUC vs Frequency Difference (Δf)")
plt.xlabel(r"Parameter Difference $(b_0 - b_1)$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
#plt.xlim(-0.05, 0.65)
plt.savefig("results/fhn_parameter/param_diff.png", dpi=180)
plt.show()



# --------- scatter plot, lim x axis
plt.figure(figsize=(10, 6))

for method, marker in markers.items():
    data = df_results[df_results["Method"] == method]
    plt.scatter(
        data["b"], data["AUC"],
        label=method,
        marker=marker,
        alpha=0.7
    )
plt.xlabel(r"Parameter Difference $(b_0 - b_1)$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
#plt.xscale('log')
plt.xlim(-0.05, 0.75)
plt.savefig("results/fhn_parameter/param_diff_lim.png", dpi=180)
plt.show()