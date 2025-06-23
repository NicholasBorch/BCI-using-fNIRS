from finger_tapping.search import _build_feature_table, _fit_gmm_metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random
import scipy.stats as stats

random.seed(42)
np.random.seed(42)

def evaluate_model(df, ctrl_mask, mask, tau, seed=42):
    """
    Fits a GMM on selected features and evaluates control classification metrics.

    Parameters:
    df : pd.DataFrame
        Feature table for one subject.
    ctrl_mask : np.ndarray
        Boolean mask indicating which rows are control trials.
    mask : list of str
        List of feature column names to include.
    tau : float
        Threshold for soft labeling (passed to _fit_gmm_metrics).
    seed : int
        Random seed for reproducibility.

    Returns:
    dict
        Evaluation metrics (e.g., sensitivity, specificity, etc.)
    """
    X = df[mask].to_numpy(float)
    X = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))
    return _fit_gmm_metrics(X, ctrl_mask, seed=seed)

def evaluate_subject(df):
    """
    Evaluates GMM performance on a single subject using multiple feature sets.

    Parameters:
    df : pd.DataFrame
        Feature table including a column 'is_control' and feature columns.

    Returns:
    pd.DataFrame
        DataFrame with one row per feature configuration, and columns for metrics.
    """
    ctrl_mask = df["is_control"].to_numpy(bool)

    baseline_mask = ['delta_mean', 'delta_variance']
    ica_mask = ['power_ic_left', 'power_ic_right']
    optimal_mask = ['abs_auc_activation', 'ic_left_delta_variance', 'ic_left_extrema_line_length', 'power_ic_right']

    baseline_result = evaluate_model(df, ctrl_mask, baseline_mask, tau=0.0, seed=42)
    ica_result = evaluate_model(df, ctrl_mask, ica_mask, tau=0.0, seed=42)
    optimal_result = evaluate_model(df, ctrl_mask, optimal_mask, tau=0.0, seed=42)

    combined_results = pd.DataFrame(
        [baseline_result, ica_result, optimal_result],
        index=["baseline", "ica", "optimal"]
    )
    return combined_results

# List of subject IDs to process
subject_ids = ['01', '02', '03', '04', '05']

# Evaluate all subjects and collect results
results = {}
all_results = pd.DataFrame()
for subject_id in subject_ids:
    df = _build_feature_table(subject_id).reset_index()
    all_results = pd.concat([all_results, df])
    results[subject_id] = evaluate_subject(df)

# Combine into multi-index DataFrame
combined_df = pd.concat(results, names=["subject", "model"])
combined_df = combined_df.reset_index()

# Summary statistics per model
combined_df.iloc[:,1:].groupby('model').mean().reset_index()
combined_df.iloc[:,1:].groupby('model').var().reset_index()

# Extract data for hypothesis testing
base = combined_df[combined_df['model'] == 'baseline']
ica = combined_df[combined_df['model'] == 'ica']

# Load 'optimal' results (presumably previously exported)
optimal_load = pd.read_csv('finger_tapping/Optimal.csv')
optimal = optimal_load.iloc[:,1:]

base_only = base.iloc[:,2:]
ica_only = ica.iloc[:,2:]
optimal_only = optimal.iloc[:,2:]

# Statistical comparisons using Mann–Whitney U test
from itertools import combinations
import pandas as pd
from scipy import stats

datasets = {"base": base_only, "ica": ica_only, "optimal": optimal_only}
metrics = base_only.columns

pvals = pd.DataFrame(index=metrics)

for (n1, d1), (n2, d2) in combinations(datasets.items(), 2):
    pvals[f"{n1}_vs_{n2}"] = [
        stats.mannwhitneyu(d1[m], d2[m], alternative="two-sided")[1] for m in metrics
    ]

print(pvals.round(3))
