from sklearn.decomposition import FastICA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_ica(X: np.ndarray, y: np.ndarray, n_components: int = 2, random_state: int = 42) -> tuple[FastICA, pd.DataFrame]:
    """Running ICA on subject data"""
    ica = FastICA(n_components = n_components, max_iter=1000, tol=0.0001, random_state = random_state)
    X_scaled = StandardScaler().fit_transform(X)
    X_ica = ica.fit_transform(X_scaled)

    ica_df = pd.DataFrame(X_ica)
    ica_df['label'] = y
    return ica, ica_df