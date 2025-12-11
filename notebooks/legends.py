import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

markers = {
    "sine": "o",
    "fhn_obs": "*",
    "fhn_dyn": "s"
}
labels = {
    "sine": "Sine",
    "fhn_obs": "FHN Observed",
    "fhn_dyn": "FHN Dynamic"
}

method_colors = {
    "raw": "C0", 
    #"pca": "C1", 
    "features": "C2", 
    "features_pca": "C3"
}

# ------------------------------
# Create custom legend handles
# ------------------------------

# Signal-type markers
signal_handles = [
    Line2D([0], [0],
           marker=markers[key],
           color="black",
           markersize=10,
           linestyle="",
           label=labels[key])
    for key in markers
]

# Method color patches
method_handles = [
    Patch(facecolor=method_colors[key],
          label=key.replace("_", " ").title())
    for key in method_colors
]

# Merge into one legend row
all_handles = signal_handles + method_handles

# ------------------------------
# Create figure
# ------------------------------
fig, ax = plt.subplots(figsize=(12, 1.2))
ax.axis("off")

legend = ax.legend(
    handles=all_handles,
    loc="center",
    frameon=False,
    ncol=len(all_handles),
    columnspacing=1.5,
    handletextpad=0.5,
    fontsize=10
)

# Save figure
fig.savefig("legend_horizontal.eps", dpi=300, bbox_inches="tight")
plt.show()
