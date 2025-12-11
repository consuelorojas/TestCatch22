import pandas as pd
import numpy as np
import pycatch22 as catch22



def extract_features(signals, return_array = False, feat = None):
    """
    Extract catch22 features from a list of time series signals.

    Parameters:
    - signals (list or np.ndarray): List of 1D time series.
    - return_array (bool): If True, return a numpy array; otherwise, return a pandas DataFrame.

    Returns:
    - pd.Dataframe or np.ndarray: Feature matrix (n_signals x n_features)
    """
    feat_rows = []
    cols_names = []
    for i, signal in enumerate(signals):
        if feat is None:
            features = catch22.catch22_all(signal, short_names=True)
            feat_rows.append(features['values'])
            cols_names = features["short_names"]
        
        else:
            row = []
            for f_name in feat:
                func = getattr(catch22, f_name)
                val = func(signal)

                row.append(val)
            feat_rows.append(row)

    # Build Dataframe
    features = pd.DataFrame(feat_rows, columns=cols_names).dropna(axis=1)

    if return_array:
        return features.values

    else:
        return features

