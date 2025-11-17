import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from sklearn.svm import SVC
from pycatch22 import catch22_all
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#plt.style.use('report.mplstyle')

# own modules
sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./features"))
sys.path.append(os.path.abspath("./preprocessing"))
from dataset import create_labeled_dataset, get_kfold_splits
from features import extract_features
from preprocessing import apply_pca


cmap = mpl.colormaps.get_cmap("coolwarm").with_extremes(under="w")
cmap.set_bad("gray")


# Sine waves parameters
fbase = 5
f1 = 5.18
nperiods = 3
npoints = 7

noise = 0.1
samples = 100

X, y = create_labeled_dataset(
    [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
    (1, 'sine', {'args': [f1, noise, npoints, nperiods]})],
    n_samples_per_class=samples
)

splits = get_kfold_splits(X, y, n_splits=5, stratified=False)

# First, for each fold, get the Catch22 feature
x_feat = []

for train_idx, test_idx in splits:
    x_feat.append(extract_features(X[train_idx]))
    # x_feat is a list of dataframes, one per fold.

# Second, apply PCA to each fold separately
x_pca = []
for frame in x_feat:
    x_pca.append(apply_pca(frame, n_components=2)[0])
    # x_pca is a list of numpy arrays, one per fold.

# Finally, we get the cross relation between the two PCA components and each feature per fold
# pearson correlation between each feature and each PCA component

corr = []
for (frame, pca_data) in zip(x_feat, x_pca):
    frame['PCA1'] = pca_data[:,0]
    frame['PCA2'] = pca_data[:,1]
    corr.append(frame.corr(method='pearson'))
    # corr is a list of dataframes, one per fold.
# Now we can plot the correlation heatmaps for each fold

mean_corr = sum(corr) / len(corr)

print(mean_corr.isnull().sum())


# sort by PCA1 and drop NaN rows/columns 
order = ['acf_timescale', 'centroid_freq', 'acf_first_min', 'ami2',
       'embedding_dist', 'stretch_decreasing', 'outlier_timing_pos',
       'high_fluctuation', 'entropy_pairs', 'trev', 'outlier_timing_neg',
       'mode_5', 'mode_10', 'stretch_high', 'transition_matrix',
       'forecast_error', 'low_freq_power', 'periodicity', 'ami_timescale',
       'whiten_timescale', 'rs_range', 'dfa']

plt.figure(figsize=(6.4, 4.8))
plt.title('Mean Correlation Heatmap Across Folds')
sns.heatmap(mean_corr, fmt=".2f", cmap=cmap, cbar=True, mask=mean_corr.isnull(), vmin=-1, vmax=1)
plt.tight_layout()
#plt.savefig('notebooks/correlation_heatmap_mean_sine.png', dpi=300)
plt.show()

# now, we get the heatmap only for PCA1 and PCA2

mean_corr_pca = mean_corr.loc[['PCA1', 'PCA2'], [col for col in mean_corr.columns if col not in ['PCA1', 'PCA2']]].T

mean_corr_pca = mean_corr_pca.sort_values(by='PCA1', ascending=False)
print(mean_corr_pca.index)

plt.figure(figsize=(6.4, 4.8))

#plt.title('Mean Correlation with PCA Components')
sns.heatmap(mean_corr_pca, fmt=".2f", cmap=cmap, cbar=False, vmin=-1, vmax=1, mask=mean_corr_pca.isnull())
plt.grid(False)
plt.tight_layout()
plt.savefig('notebooks/correlations/correlation_heatmap_mean_pca_sine.eps',format='eps',
            bbox_inches='tight', dpi=300)
plt.show()