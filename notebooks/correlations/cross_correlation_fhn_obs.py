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


## sweep configuration
b0 = 0.1

b1 = 1.0
db1 = 0.032
b12 = b1 + db1
dt = 0.1


# step to subsampling
pseudo_period = 30
npp = 10          # change the number of points per period here!
step = int(pseudo_period / npp / dt)

epsilon = 0.2
I = 0
noise = 0.1

trans = 100 # transient

samples = 80


X, y, t = create_labeled_dataset([
    (0, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
    (1, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, noise]})],
    n_samples_per_class=samples, subsample_step = step, transient = trans, return_time=True
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

# sort by PCA1 and drop NaN rows/columns
mean_corr_sorted = mean_corr.sort_values(by='PCA1', ascending=False)#.dropna(axis=1, how='all').dropna(axis=0, how='all')

plt.figure(figsize=(10, 8))
plt.title('Mean Correlation Heatmap Across Folds')
sns.heatmap(mean_corr_sorted, fmt=".2f", cmap=cmap, cbar=True, vmin=-1, vmax=1, mask=mean_corr_sorted.isnull())
plt.tight_layout()
plt.savefig('notebooks/correlation_heatmap_mean_fhn_obs.png', dpi=300)
plt.show()

# now, we get the heatmap only for PCA1 and PCA2
mean_corr_pca = mean_corr.loc[['PCA1', 'PCA2'], [col for col in mean_corr_sorted.columns if col not in ['PCA1', 'PCA2']]].T
mean_corr_pca = mean_corr_pca.sort_values(by='PCA1', ascending=False)
plt.figure(figsize=(8, 4))
plt.title('Mean Correlation with PCA Components')
sns.heatmap(mean_corr_pca, fmt=".2f", cmap=cmap, cbar=True, vmin=-1, vmax=1, mask=mean_corr_pca.isnull())
plt.tight_layout()
plt.savefig('notebooks/correlation_heatmap_mean_pca_fhn_obs.png', dpi=300)
plt.show()