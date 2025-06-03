import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../signals"))
sys.path.append(os.path.abspath("../preprocessing"))
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from dispatcher import generate_signal
from preprocessing import subsample_signals

def create_labeled_dataset(class_configs, n_samples_per_class, subsample_step = None, transient = 0, return_time = False):
    """
    Generate labeled dataset using dispatcher-based models.

    Parameters:
        class_configs: list of tuples (label, model_name, generator_args)
        n_samples_per_class: int

    Returns:
        X: np.ndarray
        y: np.ndarray
    """
    X = []
    y = []
    t_list = []
    for label, model_name, generator_args in class_configs:
        signals = generate_signal(model_name, n_samples_per_class, generator_args)

        for t, s in signals:
            if subsample_step is not None:
                s = s[transient::subsample_step]
                t = t[transient::subsample_step]
            X.append(s)
            y.append(label)
            if return_time:
                t_list.append(t)
    if return_time:
        return np.array(X), np.array(y), np.array(t_list)
    else:
        return np.array(X), np.array(y)


def get_kfold_splits(X, y, n_splits = 5, random_state = 42, stratified = True):
    """
    Generate train/test splits using K-fold cross-validation.
    Parameters:
        X: np.ndarray
        y: np.ndarray
        n_splits: int
        random_state: int
        stratified: bool
    Returns:
        list of tuples (train_indices, test_indices)
    """
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return list(kf.split(X, y))

