import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
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
# pearson correlation 
'''cross= []
for (frame, pca_data) in zip(x_feat, x_pca):
    aux1 = []
    aux2 = []
    for col in frame.columns:
        pca1 = np.cross(frame[col], pca_data[:,0])
        pca2 = np.cross(frame[col], pca_data[:,1])
        aux1.append(pca1)
        aux2.append(pca2)
    cross.append([aux1, aux2])
    # cross_corrs is a list of lists, one per fold.'''
# as we should know, the cross product is not possible between vector over 3-dimensions
# so i will use pearson correlation

corr = []
for (frame, pca_data) in zip(x_feat, x_pca):
    frame['PCA1'] = pca_data[:,0]
    frame['PCA2'] = pca_data[:,1]
    corr.append(frame.corr(method='pearson'))
    # corr is a list of dataframes, one per fold.
# Now we can plot the correlation heatmaps for each fold

mean_corr = sum(corr) / len(corr)
plt.figure(figsize=(10, 8))
plt.title('Mean Correlation Heatmap Across Folds')
sns.heatmap(mean_corr, fmt=".2f", cmap='coolwarm', cbar=True)
plt.tight_layout()
#plt.savefig('notebooks/cross_correlation_heatmap_mean.png', dpi=300)
plt.show()

'''for i, c in enumerate(corr):
    plt.figure(figsize=(10, 8))
    plt.title(f'Fold {i+1} Correlation Heatmap')
    sns.heatmap(c, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.tight_layout()
    #plt.savefig(f'notebooks/cross_correlation_heatmap_fold_{i+1}.png', dpi=300)
    plt.show()
    plt.close()
'''