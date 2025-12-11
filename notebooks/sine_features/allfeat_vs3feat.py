import os
import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import pycatch22 as catch22

plt.style.use('report.mplstyle')

# own modules
sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from dataset import create_labeled_dataset, get_kfold_splits
from classification import time_experiment, run_experiment, evaluate_single_fold, time_single_fold


cmap = mpl.colormaps.get_cmap("coolwarm").with_extremes(under="w")
cmap.set_bad("0.4")

# --- Output files
#output_dir = 'notebooks/sine_features'
#output_file_results = os.path.join(output_dir, "features_results.pkl")
#output_file_times = os.path.join(output_dir, "features_times.pkl")


# --- Simulation parameters
f0 = 5
f1 = 5.18
noise = 0.1

periods = 3
npp = 200

samples = 100

# --- Features
features = [
    'CO_f1ecac',
    'SP_Summaries_welch_rect_area_5_1',
    'SP_Summaries_welch_rect_centroid',
    'CO_FirstMin_ac',
    'FC_LocalSimple_mean3_stderr'
]

X,y = create_labeled_dataset(
    [(0, 'sine', {'args': [f0, noise, npp, periods]}),
     (1, 'sine', {'args': [f1, noise, npp, periods]})],
     n_samples_per_class=samples
)


splits = get_kfold_splits(X, y, n_splits=50, stratified=True)

all_results = []
all_times = []


auc_score = []
# only features in features vector
raw, raw_time, feat, feat_time = [], [], [], []

features = []
for signal in X:
    #feat1 = catch22.CO_f1ecac(signal)
    #feat2 = catch22.SP_Summaries_welch_rect_area_5_1(signal)
    feat3 = catch22.SP_Summaries_welch_rect_centroid(signal)
    feat4 = catch22.CO_FirstMin_ac(signal)
    feat5 = catch22.FC_LocalSimple_mean3_stderr(signal)

'''
for train_idx, test_idx in splits:
    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = SVC(probability=True, random_state=42)

    # raw clf and timing
    raw.append(evaluate_single_fold(x_train, x_test, y_train, y_test, clf))
    raw_time.append(time_single_fold(x_train, x_test, y_train, y_test, clf))


    # features

    feat1 = catch22.first1e_acf_tau()
    #results = run_experiment(X,y, splits, features=features[:i+1])
    #time = time_experiment(X,y,splits, features=features[:i+1])

    all_results.append({
        'num_feat': i+1,
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })

    all_times.append({
        'num_feat': i+1,
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })


# all features time and auc
results =  run_experiment(X,y, splits)#, features = features)
time = time_experiment(X, y, splits)#, features=features)

# save all the features
all_results.append({
    'num_feat': 22,
    'raw': results['raw'],
    'pca': results['pca'],
    'features': results['features'],
    'features_pca': results['features_pca']
})

all_times.append({
    'num_feat': 22,
    'raw': results['raw'],
    'pca': results['pca'],
    'features': results['features'],
    'features_pca': results['features_pca']
})

with open(output_file_results, 'wb') as f:
    pickle.dump(all_results, f)

with open(output_file_times, 'wb') as f:
    pickle.dump(all_times,f)

print(f"Sweep complete. Results saved to {output_file_results} and {output_file_times}")

'''