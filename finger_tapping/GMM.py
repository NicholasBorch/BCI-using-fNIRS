"""
Subject-level 3-component Gaussian-Mixture pipeline
----------------------------------------------------------------------------
For each subject this script
1. Loads pre-processed epochs via "simple_pipeline";
2. Normalises condition keys to {Control, Left, Right};
3. Fits one motor-ICA model (feature_preparation);
4. Extracts the compact epoch-level feature vector;
5. Fits a 3-component GMM on those features;
6. Identifies the cluster that mostly contains "Control" epochs: that cluster = “Control”, the other two are merged into “Task”;
7. Prints a classification report and saves the feature table.
"""

import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report

from finger_tapping.preprocessing import simple_pipeline, SUBJECTS
from finger_tapping.feature_preparation4 import (fit_motor_ica, extract_all_epoch_features, DEFAULT_SAMPLING_RATE_HZ)

# Condition-name normalisation                                       
STD_KEYS = ("Control", "Left", "Right")
PATTERN_BY_KEY  = {"Control": re.compile(r"control", re.I), "Left": re.compile(r"left", re.I), "Right": re.compile(r"right", re.I)}


def normalise_condition_dict(raw_obj) -> dict[str, "mne.Epochs"]:
    """
    Ensuring dictionary with keys {Control, Left, Right} if available.
    Works for both styles returned by "simple_pipeline".
    """
    import mne
    # Simple_pipeline can return either a dict or a single Epochs object
    if isinstance(raw_obj, dict):
        src_dict = raw_obj
    elif isinstance(raw_obj, mne.Epochs):
        src_dict = {k: raw_obj[k] for k in raw_obj.event_id}
    else:
        raise TypeError("simple_pipeline returned an unexpected type")

    normed: dict[str, mne.Epochs] = {}
    for std_key, rx in PATTERN_BY_KEY.items():
        for k, v in src_dict.items():
            if rx.search(k):
                normed[std_key] = v
                break

    if "Control" not in normed:
        raise ValueError("No control condition found for this subject")

    return normed


#  GMM helpers functions  
def fit_3comp_gmm(X_std: np.ndarray, seed: int = 42) -> GaussianMixture:
    """Return a fitted 3-component full-covariance GMM."""
    return GaussianMixture(n_components = 3, covariance_type = "full", random_state=seed, n_init=20, reg_covar=1e-6, max_iter=500).fit(X_std)


def binary_labels_from_clusters(cluster_ids: np.ndarray, control_mask: np.ndarray) -> np.ndarray:
    """
    Decide which cluster is "Control" by majority vote among known control epochs, then label:
        0 = Control-cluster
        1 = Task (the other two clusters merged)
    """
    if control_mask.shape != cluster_ids.shape or control_mask.dtype != bool:
        raise ValueError("`control_mask` must be bool array of same length")

    ctrl_ratio = [np.mean(control_mask[cluster_ids == c]) for c in range(3)]
    control_cluster = int(np.argmax(ctrl_ratio))
    return np.where(cluster_ids == control_cluster, 0, 1)


#  Feature extraction per subject                                     
def build_feature_table(subject_id: str) -> pd.DataFrame:
    """
    Return a DataFrame with indexes = epoch_id and columns:
    - Every numeric feature
    - Condition (str) - is_control (bool)
    """
    ep_raw = simple_pipeline(subject=subject_id, save = False)
    ep_dict = normalise_condition_dict(ep_raw)

    ica_model, ic_left, ic_right = fit_motor_ica(ep_dict)

    rows = []
    for cond, epochs in ep_dict.items():
        for idx in range(len(epochs)):
            epoch_data = epochs[idx].get_data(picks = ["hbo", "hbr"])[0]
            feats = extract_all_epoch_features(epoch_data, sampling_rate_hz = DEFAULT_SAMPLING_RATE_HZ, ica_model = ica_model, ic_left_index = ic_left, ic_right_index = ic_right)

            feats.update({"condition" : cond, "is_control": cond == "Control", "epoch_id"  : f"{cond}_{idx:03d}"})
            rows.append(feats)

    return pd.DataFrame(rows).set_index("epoch_id")


#  Subject-level pipeline                                             
def run_subject_pipeline(subject_id: str) -> None:
    print(f"\n==========  SUBJECT {subject_id}  ==========")
    df = build_feature_table(subject_id)

    # Preparing data
    X = df.drop(columns=["condition", "is_control"]).to_numpy(float)
    X = np.nan_to_num(X, nan=0.0)
    y_ctrl_msk = df["is_control"].to_numpy(bool)
    y_true = (~y_ctrl_msk).astype(int) # 0 = Control, 1 = Task

    X_std = StandardScaler().fit_transform(X)

    # GMM
    gmm = fit_3comp_gmm(X_std)
    raw_clusters = gmm.predict(X_std)
    y_pred = binary_labels_from_clusters(raw_clusters, y_ctrl_msk)

    # Report
    n_ctrl, n_task = y_ctrl_msk.sum(), (~y_ctrl_msk).sum()
    print(f"Epochs: total={len(df)}   control={n_ctrl}   task={n_task}\n")
    print(classification_report(y_true, y_pred,
                                target_names=["Control", "Task"]))

    # Saving feature table
    out_dir = Path("feature_tables")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"subject_{subject_id}_epoch_features.csv"
    df.to_csv(path)
    print(f"\nFeature table saved → {path.resolve()}")


if __name__ == "__main__":
    sub_code = sys.argv[1] if len(sys.argv) > 1 else SUBJECTS[0]
    run_subject_pipeline(sub_code)
