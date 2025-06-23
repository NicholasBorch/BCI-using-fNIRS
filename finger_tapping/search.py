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

N_RANDOM_TRIALS = 30_000
MAX_FEATURES    = 5
RNG             = random.Random(42)

# Helper functions
def _normalise_conditions(ep_raw) -> dict[str, "mne.Epochs"]:
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
    ep_dict = _normalise_conditions(simple_pipeline(subject, save = False))
    ica, ic_l, ic_r = fit_motor_ica(ep_dict)
    rows = []
    for cond, epochs in ep_dict.items():
        for i in range(len(epochs)):
            sig = epochs[i].get_data(picks=["hbo","hbr"])[0]
            feats = extract_all_epoch_features(sig, sampling_rate_hz = DEFAULT_SAMPLING_RATE_HZ, ica_model = ica, ic_left_index = ic_l, ic_right_index = ic_r)
            feats.update({"condition": cond, "is_control": (cond=="Control"), "epoch_id": f"{cond}_{i:03d}"})
            rows.append(feats)
    return pd.DataFrame(rows).set_index("epoch_id")

def _fit_gmm_metrics(X: np.ndarray, ctrl_mask: np.ndarray, seed: int = 42) -> dict[str, float]:
    """
    Fit a 3-component GMM and use MAP (hard) labels:
      - Identify control-cluster by majority of ctrl_mask
      - Label that cluster = 0 (control), others = 1 (task)
      - Compute F2 and other metrics
    """
    gmm = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = seed).fit(X)

    # Computing MAP cluster IDs
    clusters = gmm.predict(X)
    # Selecting control cluster by majority vote among known-control
    ctrl_vote = [np.mean(clusters[ctrl_mask] == c) for c in range(3)]
    ctrl_comp = int(np.argmax(ctrl_vote))

    # Predicting labels: 0 = Control, 1 = Task
    y_pred = np.where(clusters==ctrl_comp, 0, 1)
    y_true = (~ctrl_mask).astype(int)

    # Confusion-matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    beta = 2.0
    metrics = {
        "f2": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "precision_task": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_task": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_task": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "fp_rate_control": fp / (fp+tn+1e-9),
        "ari": adjusted_rand_score(y_true, clusters),
        "bic": gmm.bic(X),
    }
    try:
        metrics["silhouette"] = silhouette_score(X, clusters)
    except ValueError:
        metrics["silhouette"] = np.nan
    return metrics

# Main search loop
def main():
    print("\n=== Building per-subject feature tables ===")
    subj_tables = {s: _build_feature_table(s) for s in tqdm(SUBJECTS)}

    # All possible feature names
    feat_cols = [c for c in subj_tables[SUBJECTS[0]].columns if c not in ("condition","is_control")]

    # Forced subsets
    forced = []
    forced.append(("delta_mean","delta_variance"))
    forced.append(("power_ic_left","power_ic_right"))
    # Adding every 3-feature combo with the two IC powers plus one extra
    base = {"power_ic_left","power_ic_right"}
    for f in feat_cols:
        if f not in base:
            combo = tuple(sorted(base|{f}))
            forced.append(combo)

    # Building full candidate list
    candidate_sets = forced.copy()
    for _ in range(N_RANDOM_TRIALS):
        k = RNG.randint(1, min(MAX_FEATURES, len(feat_cols)))
        subset = tuple(sorted(RNG.sample(feat_cols, k)))
        candidate_sets.append(subset)

    # Evaluating each unique subset
    seen, results, all_trials = set(), [], []
    for subset in tqdm(candidate_sets):
        if subset in seen:
            continue
        seen.add(subset)

        metrics_acc = defaultdict(list)
        # Per-subject metrics
        for df in subj_tables.values():
            X = df.loc[:, subset].to_numpy(float)
            X = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))
            ctrl_mask = df["is_control"].to_numpy(bool)
            m = _fit_gmm_metrics(X, ctrl_mask)
            for k,v in m.items():
                metrics_acc[k].append(v)

        mean_m = {k: float(np.mean(vs)) for k,vs in metrics_acc.items()}
        mean_m.update({"subset": ",".join(subset), "n_feat": len(subset)})
        all_trials.append(mean_m)

        # Tracking the best subset
        if not results or mean_m["f2"] > results[0][1]["f2"] or (mean_m["f2"] == results[0][1]["f2"] and len(subset)<results[0][1]["n_feat"]):
            results = [(subset, mean_m)]

    # Saving trials and best progression
    out_dir = Path("search_results"); out_dir.mkdir(exist_ok=True)
    pd.DataFrame(all_trials).to_csv(out_dir/"all_trials.csv", index=False)

    best_subset, best_m = results[0]
    print("\n=== BEST FEATURE SET (by mean F2) ===")
    print(f" Size={len(best_subset)} → {best_subset}")
    for key in ("f2","precision_task","recall_task","f1_task",
                "accuracy","fp_rate_control","ari","bic","silhouette"):
        print(f"{key:>18s}: {best_m[key]:.3f}")
    print(f"\nAll trials → {out_dir/'all_trials.csv'}")

if __name__=="__main__":
    main()
