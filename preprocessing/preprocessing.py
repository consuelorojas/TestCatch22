import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def apply_pca(X, n_components = 2):
    """
    Apply PCA to reduce the dimensionality of the dataset.

    Parameters:
    - X (np.ndarray): Input data.
    - n_components (int): Number of principal components to keep.

    Returns:
    - np.ndarray: Transformed data with reduced dimensions.
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    elif isinstance(X, np.ndarray):
        X_values = X
    else:
        raise ValueError("Input data must be a numpy array or pandas DataFrame.")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_values)

    return X_pca, pca

# for test data, we need to apply pca.transform for the data in the folds.

def subsample_signals(signals, step =1, transient = 0):
    """
    
    Subsample signals by taking every nth sample and removing the transient part.
    
    Parameters:
    - signals (lif of np.ndarray): List of 1D time series signals.
    - step (int): Step size for subsampling.
    - transient (int): Number of initial points to skip.

    Returns:
    - list of np.ndarray: List of subsampled signals.
    """
    if step < 1:
        raise ValueError("Step size must be at least 1.")
    if transient < 0:
        raise ValueError("Transient must be non-negative.")
    
    return [signal[transient::step] for signal in signals]


