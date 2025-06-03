import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


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