import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import simple_pipeline, SUBJECTS
from ica import run_ica    

def plot_clustering(ica_df: pd.DataFrame) -> None:
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

def GMM_on_IC(subject_id: str, window_size: int = 10, n_components_ica: int = 5, sampling_rate_hz: float = 7.81, random_state: int = 42, plot: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform ICA and GMM clustering on finger tapping data for a given subject.
    
    Parameters:
    - subject: str, subject identifier
    - n_components_ica: int, number of ICA components
    - sampling_rate_hz: float, sampling rate of the data
    - plot: bool, whether to plot the results
    - random_state: int, random state for reproducibility
    """
        
    epochs = simple_pipeline(subject=subject_id)
    _, ica_df_ = run_ica(epochs, n_components=n_components_ica, random_state=random_state)
    labels = ica_df_['label']
   
    ica_df = ica_df_.iloc[:, :2].copy()
    ica_df.columns = ["IC1", "IC2"]
    ica_df["time"] = np.arange(len(ica_df)) / sampling_rate_hz    
    
    ica_df[["IC1", "IC2"]] = ica_df[["IC1", "IC2"]].rolling(window=10, min_periods=1).mean()
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gmm_labels = gmm.fit_predict(ica_df[['IC1', 'IC2']])
    ica_df["cluster"] = gmm_labels
    
    if plot:
        plot_clustering(ica_df)
    
    return ica_df, labels

def predict_p_value(subject_id: str, verbose: bool = True) -> float:
    """Predicts the p-value for the GMM clustering results on a specific subject's data.
    Args:
        subject_id (str): The identifier for the subject whose data is to be analyzed.
    Returns: p-value (float): The p-value from the chi-squared test comparing control and activity clusters.
    """
    df, labels = GMM_on_IC(subject_id=subject_id, plot=True)
    df['label'] = labels
    control_df = df[df['label'] == 'control']
    activity_df = df[df['label'] != 'control']
    
    table = np.array([
        [(control_df['cluster'] == i).sum() for i in range(len(df['cluster'].unique()))],
        [(activity_df['cluster'] == i).sum() for i in range(len(df['cluster'].unique()))]
    ])

    chi2, p, *_ = chi2_contingency(table)
    if verbose:
        print(f"Chi2: {chi2:.3f}, p-value: {p:.3g}")
        print(table)
    return p #type: ignore

def threshold_p_val(p: float, threshold: float = 0.05) -> bool:
    """Checks if the p-value is below a certain threshold.
    
    Args:
        p (float): The p-value to check.
        threshold (float): The significance threshold, default is 0.05.
        
    Returns:
        bool: True if p < threshold, False otherwise.
    """
    return p < threshold

if __name__ == "__main__":
    from colorama import Fore
    for subject in SUBJECTS:
        print(f"Processing subject {subject}...")
        p_value = predict_p_value(subject)
        print(f"Subject {subject} p-value: {p_value:.3g}")
        if threshold_p_val(p_value):
            print(Fore.GREEN + f"Subject {subject} shows significant clustering (p < 0.05).")
        else:
            print(Fore.RED + f"Subject {subject} does not show significant clustering (p >= 0.05).")
        print(Fore.RESET)