import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
plt.style.use('report.mlpstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/sine_periods/results_20250608_235320.pkl"
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

plt.figure(figsize=(10, 6))

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
plt.savefig(f"results/sine_periods/auc_vs_np_{datetime.now()}.png", dpi=180)
plt.show()
