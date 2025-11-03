import os
import sys
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('report.mplstyle')

# own modules
sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./features"))
sys.path.append(os.path.abspath("./preprocessing"))
from dataset import create_labeled_dataset, get_kfold_splits
from features import extract_features
from preprocessing import apply_pca

cmap = mpl.colormaps.get_cmap("coolwarm").with_extremes(under="w")
cmap.set_bad("0.4")

# sine
fbase = 5
f1 = 5.18

nperiods = 3
npoints = 7

# fhn obs
b_obs = 1.032

# fhn dyn
b_dyn = 1.175


b0 = 0.1
b1 = 1.0
epsilon = 0.2
I = 0
dt = 0.1

pseudo_perid = 30
npp = 10
step = int(pseudo_perid/npp/dt)
trans = 100
noise = 0.1

samples = 80

# sine
Xs, ys = create_labeled_dataset(
    [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
    (1, 'sine', {'args': [f1, noise, npoints, nperiods]})],
    n_samples_per_class=samples
)

Xobs, yobs, t = create_labeled_dataset([
    (0, 'fhn_obs', {'length':850, 'dt': dt, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
    (1, 'fhn_obs', {'length':850, 'dt': dt, 'x0': [0,0], 'args':[b0, b_obs, epsilon, I, noise]})],
    n_samples_per_class=samples, subsample_step = step, transient = trans, return_time=True
    )

Xdyn, ydyn, t = create_labeled_dataset([
    (0, 'fhn_obs', {'length':850, 'dt': dt, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
    (1, 'fhn_obs', {'length':850, 'dt': dt, 'x0': [0,0], 'args':[b0, b_dyn, epsilon, I, noise]})],
    n_samples_per_class=samples, subsample_step = step, transient = trans, return_time=True
    )

Sfeats = extract_features(Xs)
ObsFeats = extract_features(Xobs)
DynFeats = extract_features(Xdyn)

Spca = apply_pca(Sfeats, n_components=2)[0]
ObsPCA = apply_pca(ObsFeats, n_components = 2)[0]
DynPCA = apply_pca(DynFeats, n_components = 2)[0]

Sfeats['PCA1'] = Spca[:,0]
Sfeats['PCA2'] = Spca[:,1]
ObsFeats['PCA1'] = ObsPCA[:,0]
ObsFeats['PCA2'] = ObsPCA[:,1]
DynFeats['PCA1'] = DynPCA[:,0]
DynFeats['PCA2'] = DynPCA[:,1]

S_corr = Sfeats.corr(method='pearson')
Obs_corr = ObsFeats.corr(method='pearson')
Dyn_corr = DynFeats.corr(method='pearson')

order = S_corr.sort_values(by='PCA1', ascending=False).loc[lambda x: ~x.index.isin(['PCA1', 'PCA2'])].index

Splot = S_corr.loc[['PCA1', 'PCA2'], [col for col in S_corr.columns if col not in ['PCA1', 'PCA2']]]
Splot = Splot[order].T
Splot = Splot.astype(float)
Obs_plot = Obs_corr.loc[['PCA1', 'PCA2'], [col for col in Obs_corr.columns if col not in ['PCA1', 'PCA2']]]
Obs_plot = Obs_plot[order].T
Dyn_plot = Dyn_corr.loc[['PCA1', 'PCA2'], [col for col in Dyn_corr.columns if col not in ['PCA1', 'PCA2']]]
Dyn_plot = Dyn_plot[order].T


from matplotlib import gridspec

# ensure these exist: Splot, Obs_plot, Dyn_plot
# common vmin/vmax so color scale is consistent across panels
vmin, vmax = -1, 1

# Figure + GridSpec: reserve a small extra column for the colorbar
fig = plt.figure(figsize=(12, 4.2))
# 4 columns: heatmap1, heatmap2, heatmap3, colorbar
spec = gridspec.GridSpec(nrows=1, ncols=4, width_ratios=[1, 1, 1, 0.08], wspace=0.3)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
cax = fig.add_subplot(spec[3])  # colorbar axis

# common mask if you have NaNs
#mask1 = Splot.isnull() if hasattr(Splot, "isnull") else None
#mask2 = Obs_plot.isnull() if hasattr(Obs_plot, "isnull") else None
#mask3 = Dyn_plot.isnull() if hasattr(Dyn_plot, "isnull") else None

# Plot 1: left — show y tick labels
sns.heatmap(Splot, ax=ax1, cmap=cmap, vmin=-1, vmax=1, cbar=False, square=False)
ax1.set_ylabel("")   # remove default axis label if any
ax1.tick_params(left=True, labelleft=True, bottom=False, labelbottom=True)
ax1.grid(False)
# Plot 2: center — no y tick labels, no ticks
sns.heatmap(Obs_plot, ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax,
            fmt=".2f", cbar=False, square=False)
ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=True)
ax2.grid(False)


# Plot 3: right — no y tick labels, has colorbar
sns.heatmap(Dyn_plot, ax=ax3, cmap=cmap, vmin=vmin, vmax=vmax,
            fmt=".2f", cbar=True, cbar_ax=cax, square=False)
ax3.tick_params(left=False, labelleft=False, bottom=False, labelbottom=True)
ax3.grid(False)

# Labels outside (figure coordinates)
fig.text(0.2, 0.90, "(a)", fontweight="bold", fontsize=11, va="bottom", ha="left")
fig.text(0.435, 0.90, "(b)", fontweight="bold", fontsize=11, va="bottom", ha="left")
fig.text(0.67, 0.90, "(c)", fontweight="bold", fontsize=11, va="bottom", ha="left")

# Optional: tighten layout and save
plt.grid(False)
plt.subplots_adjust(left=0.20, right=0.92)# tweak if colorbar clipped
plt.savefig("notebooks/correlations/three_heatmaps_combined.eps", bbox_inches='tight', dpi=300)
plt.show()