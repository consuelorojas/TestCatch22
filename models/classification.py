import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import time

import sys
import os
sys.path.append(os.path.abspath("./preprocessing"))
sys.path.append(os.path.abspath("./features"))

from preprocessing import apply_pca #type: ignore
from features import extract_features #type: ignore

# ---------- Core classification evaluation functions ----------
def evaluate_model_auc(splits_frames, classifier, probability=True):
    """
    Evaluate a model using k-fold splits and compute AUC scores
    
    Parameters:
    - splits_frames (list): list of tuples containing (x_train, x_test, y_train, y_test) per fold
    - classifier: scikit-learn model compatible classifier
    - probability (bool): whether model supports 'predict_proba' method or to use 'decision_function'
    
    Returns:
    - auc_scores (list): AUC scores for each fold
    
    """
    auc_scores = []
    for x_train, x_test, y_train, y_test in splits_frames:
        # Ensure the classifier is fitted on the training data
        classifier.fit(x_train, y_train)

        # compute probabilities or decision function
        if probability and hasattr(classifier, 'predict_proba'):
            y_pred = classifier.predict_proba(x_test)[:, 1]
        elif hasattr(classifier, 'decision_function'):
            y_pred = classifier.decision_function(x_test)
        else:
            raise ValueError("Model does not support probability prediction or decision function.")
        
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc)
        print(f"Fold AUC: {auc:.3f}")
    return auc_scores


def evaluate_model_accuracy(splits_frames, classifier):
    """
    Evaluate a model using k-fold splits and compute accuracy scores
    
    Parameters:
    - splits_frames (list): list of tuples containing (x_train, x_test, y_train, y_test) per fold
    - classifier: scikit-learn model compatible classifier
    
    Returns:
    - accuracy_scores (list): Accuracy scores for each fold
    
    """
    accuracy_scores = []
    for x_train, x_test, y_train, y_test in splits_frames:
        # Ensure the classifier is fitted on the training data
        classifier.fit(x_train, y_train)
        
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f"Fold Accuracy: {accuracy:.3f}")
    
    return accuracy_scores

def evaluate_single_fold(X_train, X_test, y_train, y_test, classifier, probability=True):
    """
    Evaluate a single fold of data using a classifier and compute AUC score.
    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - classifier: scikit-learn compatible classifier
    - probability (bool): whether model supports 'predict_proba' method or to use 'decision_function'
    Returns:
    - auc_score: AUC score for the fold
    """
    # grid search 
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma":["scale", 0.01, 0.1, 1]
    }

    # cross validation
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    grid = GridSearchCV(classifier, param_grid, cv=cv, scoring="roc_auc" if len(np.unique(y_train)) == 2 else "accuracy")
    grid.fit(X_train, y_train)
    
    # best estimator
    classifier_tuned = grid.best_estimator_
    classifier_tuned.fit(X_train, y_train)

    # compute probabilities or decision function
    if probability and hasattr(classifier_tuned, 'predict_proba'):
        y_pred = classifier_tuned.predict_proba(X_test)[:, 1]

    elif hasattr(classifier_tuned, 'decision_function'):
        y_pred = classifier_tuned.decision_function(X_test)

    else:
        raise ValueError("Model does not support probability prediction or decision function.")
    
    return roc_auc_score(y_test, y_pred)

def time_single_fold(X_train, X_test, y_train, y_test, classifier, probability=True):
    """
    Evaluate a single fold of data using a classifier and compute AUC score.
    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - classifier: scikit-learn compatible classifier
    - probability (bool): whether model supports 'predict_proba' method or to use 'decision_function'
    Returns:
    - auc_score: AUC score for the fold
    """
    train_time, test_time = 0, 0
    start = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start
    # compute probabilities or decision function
    if probability and hasattr(classifier, 'predict_proba'):
        _= classifier.predict_proba(X_test)[:, 1]
        test_time = time.time() - start - train_time

    elif hasattr(classifier, 'decision_function'):
        _ = classifier.decision_function(X_test)
        test_time = time.time() - start - train_time

    else:
        raise ValueError("Model does not support probability prediction or decision function.")
    
    return train_time, test_time

# ---------- Pipeline function ----------

def run_experiment(X,y, splits, n_pca_components = 0.95, ffts=False, clf_fn = None, features = None, ):
    """
    Run classification pipeline with four config
    - Raw
    - Raw + PCA
    - Features
    - Features + PCA

    Parameters:
    - X (np.ndarray): raw input singals
    - y (np.ndarray): labels
    - splits (list) of (train_idx, test_idx)
    # - feature_fn (callable): function to extract features from raw signals
    - n_pca_components (int): number of PCA components to keep
    - clf_fb (callable): classifier function, if None, use default classifier

    Returns:
    - results (dict): dictionary containing AUC scores for each configuration
    """

    if clf_fn is None:
        clf_fn = lambda: SVC(probability=True, random_state=42)
    
    raw, pca, feat, feat_pca = [], [], [], []
    fft, pca_fft  = [], []

    for train_idx, test_idx  in splits:
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # raw
        clf = clf_fn()
        scaler_raw = StandardScaler()

        # scale raw features before classification
        x_train_raw = scaler_raw.fit_transform(x_train)
        x_test_raw = scaler_raw.transform(x_test)

        raw.append(evaluate_single_fold(x_train_raw, x_test_raw, y_train, y_test, clf))

        if ffts:
            # fft features
            X_fft_train = np.abs(np.fft.rfft(x_train, axis=1))
            X_fft_test = np.abs(np.fft.rfft(x_test, axis=1))

            # scale fft features before classification
            scaler_fft = StandardScaler()
            X_fft_train_scaled = scaler_fft.fit_transform(X_fft_train)
            X_fft_test_scaled = scaler_fft.transform(X_fft_test)

            fft.append(evaluate_single_fold(X_fft_train_scaled, X_fft_test_scaled, y_train, y_test, clf))

            # pca on fft features
            train_pca_fft, pca_tf, scaler = apply_pca(X_fft_train,n_components = n_pca_components)
            test_pca_fft = scaler.transform(X_fft_test)
            test_pca_fft = pca_tf.transform(test_pca_fft)
            pca_fft.append(evaluate_single_fold(train_pca_fft, test_pca_fft, y_train, y_test, clf))

        # raw + pca
        clf = clf_fn()
        train_pca, pca_tf, scaler = apply_pca(x_train,n_components = n_pca_components)
        test_pca = scaler.transform(x_test)
        test_pca = pca_tf.transform(test_pca)
        
        pca.append(evaluate_single_fold(train_pca, test_pca, y_train, y_test, clf))

        # features
        clf = clf_fn()
        if features is None:
            X_feat = extract_features(X, return_array=True)
        else:
            X_feat = extract_features(X, return_array=True, feat=features)

        # scale features before classification
        train_feat, test_feat = X_feat[train_idx], X_feat[test_idx]
        scaler_fft = StandardScaler()
        train_feat = scaler_fft.fit_transform(train_feat)
        test_feat = scaler_fft.transform(test_feat)
        
        feat.append(evaluate_single_fold(train_feat, test_feat, y_train, y_test, clf))


        # features + pca
        clf = clf_fn()
        train_feat_pca, pca_tf, scaler = apply_pca(train_feat, n_components=n_pca_components)
        test_feat = test_feat.values if isinstance(test_feat, pd.DataFrame) else test_feat
    
        test_feat_pca = scaler.transform(test_feat)
        test_feat_pca = pca_tf.transform(test_feat_pca)
        
        feat_pca.append(evaluate_single_fold(train_feat_pca, test_feat_pca, y_train, y_test, clf))

    if ffts:
        return {
            "raw": raw,
            "pca": pca,
            "features": feat,
            "features_pca": feat_pca,
            "fft": fft,
            "fft_pca": pca_fft
        }
    else:
        return {
            "raw": raw,
            "pca": pca,
            "features": feat,
            "features_pca": feat_pca
        }

# Time for the run_experiment function. Get the time per configuration

def time_experiment(X,y, splits, n_pca_components = 2, clf_fn = None, features=None, ffts=False):
    """
    Time the classification pipeline with four config
    - Raw
    - Raw + PCA
    - Features
    - Features + PCA

    Parameters:
    - X (np.ndarray): raw input singals
    - y (np.ndarray): labels
    - splits (list) of (train_idx, test_idx)
    # - feature_fn (callable): function to extract features from raw signals
    - n_pca_components (int): number of PCA components to keep
    - clf_fn (callable): classifier function, if None, use default classifier

    Returns:
    - results (dict): dictionary containing times scores for each configuration and split.
    """

    if clf_fn is None:
        clf_fn = lambda: SVC(probability=True, random_state=42)
    
    raw, pca, feat, feat_pca = [], [], [], []
    fft, pca_fft  = [], []

    for train_idx, test_idx  in splits:
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # scale raw features before classification


        # raw
        clf = clf_fn()
        time_raw = time.time()
        scaler_raw = StandardScaler()
        x_train_raw = scaler_raw.fit_transform(x_train)
        x_test_raw = scaler_raw.transform(x_test)
        time_raw = time.time() - time_raw
        train_time, test_time = time_single_fold(x_train_raw, x_test_raw, y_train, y_test, clf)
        raw.append((train_time, test_time, time_raw))

        # raw + pca
        clf = clf_fn()
        pca_time = time.time()
        train_pca, pca_tf, scaler = apply_pca(x_train,n_components = 1)

        test_pca = scaler.transform(x_test)
        test_pca = pca_tf.transform(test_pca)
        pca_time = time.time() - pca_time

        train_time, test_time = time_single_fold(train_pca, test_pca, y_train, y_test, clf)
        pca.append((train_time, test_time, pca_time))
        
        # fft features
        if ffts:
            # fft features
            clf = clf_fn()
            fft_time = time.time()
            X_fft_train = np.abs(np.fft.rfft(x_train, axis=1))
            X_fft_test = np.abs(np.fft.rfft(x_test, axis=1))

            scaler_fft = StandardScaler()
            X_fft_train_scaled = scaler_fft.fit_transform(X_fft_train)
            X_fft_test_scaled = scaler_fft.transform(X_fft_test)
            fft_time = time.time() - fft_time

            train_time, test_time = time_single_fold(X_fft_train_scaled, X_fft_test_scaled, y_train, y_test, clf)
            fft.append((train_time, test_time, fft_time))

            clf = clf_fn()
            fft_time = time.time()

            train_fft, pca_tf_fft, scaler_fft = apply_pca(X_fft_train,n_components = 2)
            test_fft = scaler_fft.transform(X_fft_test)
            test_fft = pca_tf_fft.transform(test_fft)
            fft_time = time.time() - fft_time

            train_time, test_time = time_single_fold(train_fft, test_fft, y_train, y_test, clf)
            pca_fft.append((train_time, test_time, fft_time))

        # features
        clf = clf_fn()

        feat_time = time.time()
        X_feat = extract_features(X, return_array=True)
        train_feat, test_feat = X_feat[train_idx], X_feat[test_idx]
        scaler_fft = StandardScaler()
        train_feat = scaler_fft.fit_transform(train_feat)
        test_feat = scaler_fft.transform(test_feat)

        feat_time = time.time() - feat_time

        train_time, test_time = time_single_fold(train_feat, test_feat, y_train, y_test, clf)
        feat.append((train_time, test_time, feat_time))


        # features + pca
        clf = clf_fn()
        feat_pca_time = time.time()
        train_feat_pca, pca_tf, scaler = apply_pca(train_feat, n_components=n_pca_components)
        test_feat = test_feat.values if isinstance(test_feat, pd.DataFrame) else test_feat
    
        test_feat_pca = scaler.transform(test_feat)
        test_feat_pca = pca_tf.transform(test_feat_pca)
        feat_pca_time = time.time() - feat_pca_time
        train_time, test_time = time_single_fold(train_feat_pca, test_feat_pca, y_train, y_test, clf)

        feat_pca.append([train_time, test_time, feat_pca_time+feat_time])
    if ffts:
        return {
            "raw": raw,
            "pca": pca,
            "features": feat,
            "features_pca": feat_pca,
            "fft": fft,
            "fft_pca": pca_fft
            }
    else:
        return {
            "raw": raw,
            "pca": pca,
            "features": feat,
            "features_pca": feat_pca}
