import pytest
import numpy as np
import pandas as pd

from finger_tapping.pca_ica import run_pca
from finger_tapping.preprocessing import simple_pipeline

@pytest.fixture(scope="module")
def subject_epochs():
    return simple_pipeline(subject="01", save=False)

def test_run_pca_output(subject_epochs):
    pca, pca_df, X_scaled, y = run_pca(subject_epochs, n_components=3, random_state=0)

    # DataFrame is non-empty
    assert isinstance(pca_df, pd.DataFrame)
    assert not pca_df.empty
    
    # Scaled X is a non-empty numpy array
    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.size > 0
    
    # y is a non-empty numpy array
    assert isinstance(y, np.ndarray)
    assert y.size > 0
