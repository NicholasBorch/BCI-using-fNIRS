import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from feature_preparation import extract_X_y
from preprocessing import simple_pipeline
from pca_ica import run_ica

def simple_GMM(subject_id: str, n_components_ica: int = 5, sampling_rate_hz: float = 7.81, plot: bool = True, random_state: int = 42):
    """
    Perform ICA and GMM clustering on finger tapping data for a given subject.
    
    Parameters:
    - subject: str, subject identifier
    - n_components_ica: int, number of ICA components
    - sampling_rate_hz: float, sampling rate of the data
    - plot: bool, whether to plot the results
    - random_state: int, random state for reproducibility
    """
        
    subject = simple_pipeline(subject=subject_id)
    ica, ica_df_ = run_ica(subject, n_components=n_components_ica, random_state=random_state)
   
    ica_df = ica_df_.iloc[:, :2].copy()
    ica_df.columns = ["IC1", "IC2"]
    ica_df["time"] = np.arange(len(ica_df)) / sampling_rate_hz

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gmm_labels = gmm.fit_predict(ica_df[['IC1', 'IC2']])
    ica_df["cluster"] = gmm_labels

    if plot:

        # --- Plot IC1 over time ---
        plt.figure(figsize=(12, 4))
        sns.scatterplot(data=ica_df, x='time', y='IC1', hue='cluster', palette='Set2', s=10)
        plt.axvline(x=ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 1')
        plt.axvline(x=2*ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 2')
        plt.title("IC1 over Time with GMM Clusters")
        plt.xlabel("Time (s)")
        plt.ylabel("IC1 Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot IC2 over time ---
        plt.figure(figsize=(12, 4))
        sns.scatterplot(data=ica_df, x='time', y='IC2', hue='cluster', palette='Set2', s=10)
        plt.axvline(x=ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 1')
        plt.axvline(x=2*ica_df.time[len(ica_df)//3], color='red', linestyle='--', label='Cue 2')
        plt.title("IC2 over Time with GMM Clusters")
        plt.xlabel("Time (s)")
        plt.ylabel("IC2 Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return ica, ica_df, gmm



###
if __name__ == "__main__":
    simple_GMM("01", plot=True)


