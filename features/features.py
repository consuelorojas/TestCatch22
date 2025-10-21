import pandas as pd
import numpy as np
from pycatch22 import catch22_all



def extract_features(signals, return_array = False):
    """
    Extract catch22 features from a list of time series signals.

    Parameters:
    - signals (list or np.ndarray): List of 1D time series.
    - return_array (bool): If True, return a numpy array; otherwise, return a pandas DataFrame.

    Returns:
    - pd.Dataframe or np.ndarray: Feature matrix (n_signals x n_features)
    """
    feat_rows = []
    cols_nameEs = []
    for i, signal in enumerate(signals):
        features = catch22_all(signal, short_names=True)
        feat_rows.append(features['values'])
        if i == 0:
            cols_names = features['short_names']  


    features = np.array(feat_rows)

    if return_array:
        return features
    
    else:
        return pd.DataFrame(features, columns=cols_names)
    

