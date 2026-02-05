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

fbase = 5
deltaf = 0.01
f1 = [fbase + deltaf*i for i in range(0, 52, 2)]
dfreq = [deltaf *  i for i in range(0, 52, 2)]

npp = 10
periods = 3

noise = 0.1
samples = 100

sweep_name = "sine/freq_fft"
output_dir = os.path.join("FFTs", sweep_name)
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"sine_fft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")


all_results = []
for i, freq in enumerate(tqdm(f1, desc="Sweeping frequency differences")):
    X, y = create_labeled_dataset( # type: ignore
        [(0, 'sine', {'args': [fbase, 0.1, npp, periods]}),
         (1, 'sine', {'args': [freq, 0.1, npp, periods]})],
        n_samples_per_class=samples
    )

    # compute one-dimensional discrete fourier transform of the matrix X with real inputs
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
    all_results.append({
        'df': round(dfreq[i], 3),
        'auc': auc_scores
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")
