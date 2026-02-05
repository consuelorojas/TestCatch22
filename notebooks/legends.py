import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def export_legend(legend, filename="legend_horizontal_3.eps"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


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
handles = signal_handles + method_handles

# ------------------------------
# Create legend figure
# ------------------------------
fig, ax = plt.subplots(figsize=(12, 1))
ax.axis("off")

legend = ax.legend(
    handles=handles,
    loc="center",
    frameon=False,
    ncol=len(handles),
    columnspacing=1.0,
    handletextpad=0.35,
    fontsize=9
)

# ------------------------------
# Export tightly (legend bbox)
# ------------------------------
export_legend(legend, "legend_horizontal_all.eps")

plt.show()