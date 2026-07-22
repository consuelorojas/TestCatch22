import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.style.use("report.mplstyle")

sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./preprocessing"))

from dataset import create_labeled_dataset, get_kfold_splits
from preprocessing import apply_pca


# =============================================================================
# SINE WAVES
# =============================================================================

# Sweep configuration
fbase = 5.0
f1 = 5.4
nperiods = 3
npoints = 7

noise = 0.1
samples = 150

# =============================================================================
# Generate dataset
# =============================================================================

X, y = create_labeled_dataset(
    [
        (0, "sine", {"args": [fbase, noise, npoints, nperiods]}),
        (1, "sine", {"args": [f1, noise, npoints, nperiods]}),
    ],
    n_samples_per_class=samples,
)

# =============================================================================
# Train/Test split
# =============================================================================

splits = get_kfold_splits(X, y, n_splits=1, stratified=True)

train_idx, test_idx = splits[0]

x_train = X[train_idx]
y_train = y[train_idx]

# =============================================================================
# PCA
# =============================================================================

# Force 3 principal components for the 3D visualization
train_pca, pca_tf, scaler = apply_pca(x_train, n_components=3)

print(f"Number of PCA components: {pca_tf.n_components_}")
print(f"Explained variance ratio: {pca_tf.explained_variance_ratio_}")
print(f"Total explained variance: {pca_tf.explained_variance_ratio_.sum():.3f}")

# =============================================================================
# Plot settings
# =============================================================================

class_colors = {
    0: "tab:blue",
    1: "tab:orange",
}

labels = {
    0: rf"Class 0 ($\nu_0={fbase:.1f}$ Hz)",
    1: rf"Class 1($\nu={f1:.1f}$ Hz)",
}

pc_var = 100 * pca_tf.explained_variance_ratio_

# =============================================================================
# 2D PCA (PC1 vs PC2)
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 8))

for cls in np.unique(y_train):
    mask = y_train == cls

    ax.scatter(
        train_pca[mask, 0],
        train_pca[mask, 1],
        color=class_colors[cls],
        s=60,
        edgecolor="k",
        linewidth=0.4,
        label=labels[cls],
    )

ax.set_xlabel(f"PC1 ({pc_var[0]:.1f}% explained variance)")
ax.set_ylabel(f"PC2 ({pc_var[1]:.1f}% explained variance)")

ax.set_title(
    rf"Parameter difference: $(\nu-\nu_0) = {f1-fbase:.2f}\,\mathrm{{Hz}}$"
)

ax.legend(frameon=True)

plt.tight_layout()
plt.savefig(
    f"raw_pca_2d_{f1-fbase:.2f}.eps",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# =============================================================================
# 3D PCA (PC1 vs PC2 vs PC3)
# =============================================================================

fig = plt.figure(figsize=(9, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection="3d")


for cls in np.unique(y_train):
    mask = y_train == cls

    ax.scatter(
        train_pca[mask, 0],
        train_pca[mask, 1],
        train_pca[mask, 2],
        color=class_colors[cls],
        s=45,
        edgecolor="k",
        linewidth=0.3,
        label=labels[cls],
    )

ax.set_xlabel(f"PC1 ({pc_var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({pc_var[1]:.1f}%)")
ax.set_zlabel(f"PC3 ({pc_var[2]:.1f}%)", labelpad=12)

plt.savefig(
    f"raw_pca_3d_{f1-fbase:.2f}.eps",
    dpi=300,
    bbox_inches="tight",
)
ax.set_title(
    rf"Parameter difference: $(\nu-\nu_0) = {f1-fbase:.2f}\,\mathrm{{Hz}}$"
)

ax.legend(frameon=True)

# Nice viewing angle
ax.view_init(elev=25, azim=-55)

plt.tight_layout()
plt.savefig(
    f"raw_pca_3d_{f1-fbase:.2f}.eps",
    dpi=300,
    bbox_inches="tight",
)
plt.show()