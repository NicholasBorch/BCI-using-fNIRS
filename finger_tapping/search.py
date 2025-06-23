#!/usr/bin/env python
"""
Random-search for the best feature subset (≤ 5 features) under MAP (hard-assignment) GMM labels.

We first force-evaluate:
  1. ["delta_mean","delta_variance"]
  2. ["power_ic_left","power_ic_right"]
  3. ["power_ic_left","power_ic_right"] + each other single feature

Then we perform N_RANDOM_TRIALS random subsets of size up to MAX_FEATURES.
Goal is to maximise mean F2 score across subjects.
"""
from pathlib import Path
import random, re
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, adjusted_rand_score, silhouette_score

from finger_tapping.preprocessing import simple_pipeline, SUBJECTS
from finger_tapping.feature_preparation import extract_all_epoch_features, fit_motor_ica, DEFAULT_SAMPLING_RATE_HZ
#!/usr/bin/env python

# Constants
N_RANDOM_TRIALS = 30_000
MAX_FEATURES    = 5
RNG             = random.Random(42)

# Helper functions
def _normalise_conditions(ep_raw) -> dict[str, "mne.Epochs"]:
    """
    Maps raw MNE event labels to standardized conditions ("Control", "Left", "Right").

    Parameters:
    ep_raw : mne.Epochs or dict
        Raw epochs object or dictionary of condition-labeled Epochs.

    Returns:
    dict
        Dictionary mapping standardized condition labels to MNE Epochs.
    
    Raises:
    RuntimeError
        If "Control" condition is not found.
    """
    import mne
    patt = {"Control": re.compile("control", re.I), "Left" : re.compile("left", re.I), "Right" : re.compile("right", re.I)}
    if isinstance(ep_raw, dict):
        ep_dict = ep_raw
    elif isinstance(ep_raw, mne.Epochs):
        ep_dict = {n: ep_raw[n] for n in ep_raw.event_id}
    else:
        raise TypeError("simple_pipeline returned unexpected object")
    out = {}
    for std, rx in patt.items():
        for k, v in ep_dict.items():
            if rx.search(k):
                out[std] = v
                break
    if "Control" not in out:
        raise RuntimeError("Control condition missing")
    return out

def _build_feature_table(subject: str) -> pd.DataFrame:
    """
    Builds a feature table for a given subject.

    Parameters:
    subject : str
        Subject ID.

    Returns:
    pd.DataFrame
        Feature table with one row per epoch and columns for extracted features,
        condition labels, and control indicators.
    """
    ep_dict = _normalise_conditions(simple_pipeline(subject, save=False))
    ica, ic_l, ic_r = fit_motor_ica(ep_dict)
    rows = []
    for cond, epochs in ep_dict.items():
        for i in range(len(epochs)):
            sig = epochs[i].get_data(picks=["hbo", "hbr"])[0]
            feats = extract_all_epoch_features(sig, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, ica_model=ica, ic_left_index=ic_l, ic_right_index=ic_r)
            feats.update({
                "condition": cond,
                "is_control": (cond == "Control"),
                "epoch_id": f"{cond}_{i:03d}"
            })
            rows.append(feats)
    return pd.DataFrame(rows).set_index("epoch_id")

def _fit_gmm_metrics(X: np.ndarray, ctrl_mask: np.ndarray, seed: int = 42) -> dict[str, float]:
    """
    Fits a 3-component GMM and computes classification metrics using MAP assignments.

    Parameters:
    X : np.ndarray
        Feature matrix (n_samples × n_features).
    ctrl_mask : np.ndarray
        Boolean mask indicating which samples are control trials.
    seed : int
        Random seed for reproducibility.

    Returns:
    dict
        Dictionary of performance metrics (F2, accuracy, precision, recall, etc.).
    """
    gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=20, max_iter=500,
                          reg_covar=1e-6, random_state=seed).fit(X)
    clusters = gmm.predict(X)
    ctrl_vote = [np.mean(clusters[ctrl_mask] == c) for c in range(3)]
    ctrl_comp = int(np.argmax(ctrl_vote))

    y_pred = np.where(clusters == ctrl_comp, 0, 1)
    y_true = (~ctrl_mask).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    beta = 2.0
    metrics = {
        "f2": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "precision_task": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_task": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_task": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "fp_rate_control": fp / (fp + tn + 1e-9),
        "ari": adjusted_rand_score(y_true, clusters),
        "bic": gmm.bic(X),
    }
    try:
        metrics["silhouette"] = silhouette_score(X, clusters)
    except ValueError:
        metrics["silhouette"] = np.nan
    return metrics

# Main execution logic
def main():
    """
    Main loop for running feature subset search and evaluating GMM classification metrics.

    - Builds feature tables for all subjects.
    - Constructs candidate feature subsets (forced + random).
    - Evaluates each subset by averaging metrics across all subjects.
    - Tracks and prints the best feature subset by mean F2 score.
    - Saves all evaluations to 'search_results/all_trials.csv'.
    """
    print("\n=== Building per-subject feature tables ===")
    subj_tables = {s: _build_feature_table(s) for s in tqdm(SUBJECTS)}

    feat_cols = [c for c in subj_tables[SUBJECTS[0]].columns if c not in ("condition", "is_control")]

    # Forced subset evaluation
    forced = [
        ("delta_mean", "delta_variance"),
        ("power_ic_left", "power_ic_right")
    ]
    base = {"power_ic_left", "power_ic_right"}
    for f in feat_cols:
        if f not in base:
            combo = tuple(sorted(base | {f}))
            forced.append(combo)

    # Generate random candidate feature subsets
    candidate_sets = forced.copy()
    for _ in range(N_RANDOM_TRIALS):
        k = RNG.randint(1, min(MAX_FEATURES, len(feat_cols)))
        subset = tuple(sorted(RNG.sample(feat_cols, k)))
        candidate_sets.append(subset)

    # Evaluate all unique candidate subsets
    seen, results, all_trials = set(), [], []
    for subset in tqdm(candidate_sets):
        if subset in seen:
            continue
        seen.add(subset)

        metrics_acc = defaultdict(list)
        for df in subj_tables.values():
            X = df.loc[:, subset].to_numpy(float)
            X = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))
            ctrl_mask = df["is_control"].to_numpy(bool)
            m = _fit_gmm_metrics(X, ctrl_mask)
            for k, v in m.items():
                metrics_acc[k].append(v)

        mean_m = {k: float(np.mean(vs)) for k, vs in metrics_acc.items()}
        mean_m.update({"subset": ",".join(subset), "n_feat": len(subset)})
        all_trials.append(mean_m)

        if not results or mean_m["f2"] > results[0][1]["f2"] or (
            mean_m["f2"] == results[0][1]["f2"] and len(subset) < results[0][1]["n_feat"]
        ):
            results = [(subset, mean_m)]

    # Save results
    out_dir = Path("search_results")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(all_trials).to_csv(out_dir / "all_trials.csv", index=False)

    # Print best subset
    best_subset, best_m = results[0]
    print("\n=== BEST FEATURE SET (by mean F2) ===")
    print(f" Size={len(best_subset)} → {best_subset}")
    for key in ("f2", "precision_task", "recall_task", "f1_task",
                "accuracy", "fp_rate_control", "ari", "bic", "silhouette"):
        print(f"{key:>18s}: {best_m[key]:.3f}")
    print(f"\nAll trials → {out_dir/'all_trials.csv'}")

if __name__ == "__main__":
    main()
