import numpy as np
from typing import Dict
from scipy.stats import ttest_ind, mannwhitneyu

from simple_GMM import simple_GMM


def evaluate_subject(subject_id: str, *, rest_fraction: float = 1 / 3, alpha: float = 0.05, verbose: bool = True, **gmm_kwargs,) -> Dict[str, float]:
    """
    Evaluate cross-cluster mean‐difference on IC2 between REST and TASK.

    Returns a dict with:
      - mean_rest_c0, mean_rest_c1, mean_task_c0, mean_task_c1
      - t_p_rest0_task0, u_p_rest0_task0
      - t_p_rest1_task1, u_p_rest1_task1
    """
    # 1) Run simple_GMM (no plotting)
    _, ica_df, _ = simple_GMM(subject_id, plot=False, **gmm_kwargs)

    ic2            = ica_df["IC2"].to_numpy()
    cluster_labels = ica_df["cluster"].to_numpy()

    # 2) Build REST / TASK masks
    n_samples = ic2.size
    split_idx = int(n_samples * rest_fraction)

    rest_mask = np.zeros(n_samples, dtype=bool)
    task_mask = np.zeros(n_samples, dtype=bool)
    rest_mask[:split_idx] = True
    task_mask[split_idx:] = True

    # 3) Extract IC2 values for each (cluster, cue) combination
    rest_c0_vals  = np.abs(ic2[(cluster_labels == 0) & rest_mask])
    rest_c1_vals  = np.abs(ic2[(cluster_labels == 1) & rest_mask])
    task_c0_vals  = np.abs(ic2[(cluster_labels == 0) & task_mask])
    task_c1_vals  = np.abs(ic2[(cluster_labels == 1) & task_mask])

    # Means
    mean_rest_c0 = rest_c0_vals.mean() if rest_c0_vals.size > 0 else np.nan
    mean_rest_c1 = rest_c1_vals.mean() if rest_c1_vals.size > 0 else np.nan
    mean_task_c0 = task_c0_vals.mean() if task_c0_vals.size > 0 else np.nan
    mean_task_c1 = task_c1_vals.mean() if task_c1_vals.size > 0 else np.nan

    # 4a) Compare REST cluster 0 → TASK cluster 0
    t_stat_00, p_t_00 = ttest_ind(rest_c0_vals, task_c0_vals, equal_var=False)
    U_stat_00, p_u_00 = mannwhitneyu(rest_c0_vals, task_c0_vals, alternative="two-sided")

    # 4b) Compare REST cluster 1 → TASK cluster 1
    t_stat_11, p_t_11 = ttest_ind(rest_c1_vals, task_c1_vals, equal_var=False)
    U_stat_11, p_u_11 = mannwhitneyu(rest_c1_vals, task_c1_vals, alternative="two-sided")

    results = {
        "mean_rest_c0": mean_rest_c0,
        "mean_rest_c1": mean_rest_c1,
        "mean_task_c0": mean_task_c0,
        "mean_task_c1": mean_task_c1,
        "t_p_rest0_task0": p_t_00,
        "u_p_rest0_task0": p_u_00,
        "t_p_rest1_task1": p_t_11,
        "u_p_rest1_task1": p_u_11,
    }

    if verbose:
        print(f"\nSubject {subject_id}")
        print(f"  Mean IC2 (REST,  cluster=0) : {mean_rest_c0:.3f}")
        print(f"  Mean IC2 (REST,  cluster=1) : {mean_rest_c1:.3f}")
        print(f"  Mean IC2 (TASK,  cluster=0) : {mean_task_c0:.3f}")
        print(f"  Mean IC2 (TASK,  cluster=1) : {mean_task_c1:.3f}\n")

        print("Comparison: REST(cluster=0)  vs  TASK(cluster=0)")
        print(f"  t-test    p = {p_t_00:.3g}  {'*' if p_t_00 < alpha else ''}")
        print(f"  Mann–Whitney p = {p_u_00:.3g}  {'*' if p_u_00 < alpha else ''}\n")

        print("Comparison: REST(cluster=1)  vs  TASK(cluster=1)")
        print(f"  t-test    p = {p_t_11:.3g}  {'*' if p_t_11 < alpha else ''}")
        print(f"  Mann–Whitney p = {p_u_11:.3g}  {'*' if p_u_11 < alpha else ''}\n")

    return results


if __name__ == "__main__":
    stats = evaluate_subject("01", verbose=True)
    print(stats)