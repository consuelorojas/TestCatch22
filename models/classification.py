import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("../preprocessing"))
sys.path.append(os.path.abspath("../features"))

from preprocessing import apply_pca
from features import extract_features

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
    classifier.fit(X_train, y_train)
    # compute probabilities or decision function
    if probability and hasattr(classifier, 'predict_proba'):
        y_pred = classifier.predict_proba(X_test)[:, 1]

    elif hasattr(classifier, 'decision_function'):
        y_pred = classifier.decision_function(X_test)

    else:
        raise ValueError("Model does not support probability prediction or decision function.")
    
    return roc_auc_score(y_test, y_pred)

# ---------- Pipeline function ----------

def run_experiment(X,y, splits, n_pca_components = 2, clf_fn = None):
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

    for train_idx, test_idx  in splits:
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # raw
        clf = clf_fn()
        raw.append(evaluate_single_fold(x_train, x_test, y_train, y_test, clf))

        # raw + pca
        clf = clf_fn()
        train_pca, pca_tf, scaler = apply_pca(x_train,n_components = n_pca_components)
        test_pca = scaler.transform(x_test)
        test_pca = pca_tf.transform(test_pca)
        
        pca.append(evaluate_single_fold(train_pca, test_pca, y_train, y_test, clf))

        # features
        clf = clf_fn()
        X_feat = extract_features(X, return_array=True)
        train_feat, test_feat = X_feat[train_idx], X_feat[test_idx]
        feat.append(evaluate_single_fold(train_feat, test_feat, y_train, y_test, clf))

        # features + pca
        clf = clf_fn()
        train_feat_pca, pca_tf, scaler = apply_pca(train_feat, n_components=n_pca_components)
        test_feat = test_feat.values if isinstance(test_feat, pd.DataFrame) else test_feat
        test_feat_pca, _, _ = apply_pca(test_feat, n_components=n_pca_components)
        #test_feat_pca = scaler.transform(test_feat)
        #test_feat_pca = pca_tf.transform(test_feat_pca)
        feat_pca.append(evaluate_single_fold(train_feat_pca, test_feat_pca, y_train, y_test, clf))

    return {
        "raw": raw,
        "pca": pca,
        "features": feat,
        "features_pca": feat_pca
    }