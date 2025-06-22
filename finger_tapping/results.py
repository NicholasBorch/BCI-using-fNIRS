from finger_tapping.search import _build_feature_table, _fit_gmm_metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import random
from finger_tapping.preprocessing import simple_pipeline


from scipy.stats import wilcoxon

random.seed(42)
np.random.seed(42)

def evaluate_model(df, ctrl_mask, mask, tau, seed=42):
    X = df[mask].to_numpy(float)
    X = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))
    return _fit_gmm_metrics(X, ctrl_mask, tau=tau, seed=seed)


def evaluate_subject(df):
    ctrl_mask = df["is_control"].to_numpy(bool)

    baseline_mask = ['delta_mean', 'delta_variance']
    ica_mask = ['power_ic_left', 'power_ic_right']
    optimal_mask = ['abs_auc_activation', 'ic_left_delta_variance', 'ic_left_extrema_line_length', 'power_ic_right']
    baseline_result = evaluate_model(df, ctrl_mask, baseline_mask, tau=0.0, seed=42)
    ica_result = evaluate_model(df, ctrl_mask, ica_mask, tau=0.0, seed=42)
    optimal_result = evaluate_model(df, ctrl_mask, optimal_mask, tau=0.8, seed=42)

    combined_results = pd.DataFrame(
        [baseline_result, ica_result, optimal_result],
        index=["baseline", "ica", "optimal"]
    )
    return combined_results

# List of subjects
subject_ids = ['01', '02', '03', '04', '05']

# Evaluate all and collect in a dict
results = {}
all_results = pd.DataFrame()
for subject_id in subject_ids:
    df = _build_feature_table(subject_id).reset_index()
    all_results = pd.concat([all_results,df])
    results[subject_id] = evaluate_subject(df)

# Combine all results into a single multi-index DataFrame
combined_df = pd.concat(results, names=["subject", "model"])
combined_df = combined_df.reset_index()

combined_df.iloc[:,1:].groupby('model').mean().reset_index()
combined_df.iloc[:,1:].groupby('model').var().reset_index()

base = combined_df[combined_df['model'] == 'baseline']
ica = combined_df[combined_df['model'] == 'ica']
optimal = combined_df[combined_df['model'] == 'optimal']


# u, p = wilcoxon(x,y, alternative="two-sided")

base_only = base.iloc[:,2:]
ica_only = ica.iloc[:,2:]
optimal_only = optimal.iloc[:,2:]
for i in range(base_only.shape[1]):
    x = base_only.iloc[:,i]
    y = optimal_only.iloc[:,i]
    _, p = wilcoxon(x,y, alternative="two-sided")
    print(p)
    
    

# control = simple_pipeline("01")["Control"].get_data()
# control_baseline = control[:,:,:39]
# control_activation = control[:,:,39:]

# abs(control_activation.mean()) - abs(control_baseline.mean())