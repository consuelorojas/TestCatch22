import os
import sys
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from dataset import create_labeled_dataset, get_kfold_splits


## sweep configuration
fbase = 5
f1 = 5.18
nperiods = 3
npoints = 7

#noise = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
noise = np.linspace(0, 0.4, 20)
bignoise = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
noise = np.concatenate((noise, bignoise))
noise = np.round(noise, 3)

samples = 100

## Output directory
sweep_name = "sine/noise_fft"
output_dir = os.path.join("FFTs", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"sine_fft_{timestamp}.pkl")

# Run sweep
all_results = []
for i, n in enumerate(tqdm(noise,desc="Sweeping noise levels")):
    X, y = create_labeled_dataset( #type: ignore
        [(0, 'sine', {'args': [fbase, n, npoints, nperiods]}),
         (1, 'sine', {'args': [f1, n, npoints, nperiods]})],
        n_samples_per_class=samples
    )

    X_fft = np.abs(np.fft.rfft(X, axis=1))
    splits = get_kfold_splits(X_fft, y, n_splits=50, stratified=True)

    auc_scores = []
    for train_idx, test_idx in splits:

        x_train, x_test = X_fft[train_idx], X_fft[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = SVC(probability=True, random_state=42)
        clf.fit(x_train, y_train)

        y_pred = clf.predict_proba(x_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, y_pred))
        #print(f"npp: {npp}, Fold AUC: {auc_scores[-1]:.3f}")
    all_results.append({
        'noise': n,
        'auc': auc_scores
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")

