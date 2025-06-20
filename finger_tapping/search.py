"""
Random-search for the best feature subset (≤ 10 features) and the best soft-decision threshold τ ∈ {0.0, 0.6, 0.8}.
-----------------------------------------------------------------------------------------------------------------------
Goal is to maximise mean F2 score across subjects  

Fixed subsets evaluated first
-------------------------------------------------------------------------------
1. All IC-derived features (``ic_*`` + ``power_ic_*``)
2. IC powers only (``power_ic_left``, ``power_ic_right``)
3. Raw-epoch power only (``power_epoch``)
4. All raw-trace features (no IC features)

Then "N_RANDOM_TRIALS" random subsets (size ≤ 10) are evaluated.
"""

from pathlib import Path
import random, re
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (fbeta_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, adjusted_rand_score, silhouette_score)

from finger_tapping.preprocessing import simple_pipeline, SUBJECTS
from finger_tapping.feature_preparation4 import (extract_all_epoch_features, fit_motor_ica, DEFAULT_SAMPLING_RATE_HZ)

# Search parameters
N_RANDOM_TRIALS = 10
MAX_FEATURES    = 5
RNG             = random.Random(42)
TAU_VALUES      = (0.0, 0.6, 0.8)   # 0.0 = hard assignment

# Helper functions
def _normalise_conditions(ep_raw) -> dict[str, "mne.Epochs"]:
    """Map arbitrary condition names --> {Control, Left, Right}."""
    import mne
    patt = {"Control": re.compile("control", re.I), "Left": re.compile("left", re.I), "Right": re.compile("right", re.I)}
    if isinstance(ep_raw, dict):
        ep_dict = ep_raw
    elif isinstance(ep_raw, mne.Epochs):
        ep_dict = {n: ep_raw[n] for n in ep_raw.event_id}
    else:
        raise TypeError("simple_pipeline returned unexpected object")

    out: dict[str, mne.Epochs] = {}
    for std, p in patt.items():
        for k, v in ep_dict.items():
            if p.search(k):
                out[std] = v
                break
    if "Control" not in out:
        raise RuntimeError("Control condition missing")
    return out


def _build_feature_table(subj_code: str) -> pd.DataFrame:
    """Return one DataFrame (rows = epochs) for a subject."""
    ep_dict = _normalise_conditions(simple_pipeline(subj_code, save = False))
    ica, ic_l, ic_r = fit_motor_ica(ep_dict)

    rows: list[dict[str, float]] = []
    for cond, epochs in ep_dict.items():
        for i in range(len(epochs)):
            sig = epochs[i].get_data(picks = ["hbo", "hbr"])[0]
            feats = extract_all_epoch_features(sig, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, ica_model=ica, ic_left_index=ic_l, ic_right_index=ic_r)
            feats |= {"condition" : cond, "is_control": cond == "Control", "epoch_id"  : f"{cond}_{i:03d}"}
            rows.append(feats)
    return pd.DataFrame(rows).set_index("epoch_id")


#  GMM fit
def _fit_gmm_metrics(X: np.ndarray, ctrl_mask: np.ndarray, tau: float, seed: int = 42) -> dict[str, float]:
    """
    - Fit 3-component full-cov GMM
    - Pick the component that contains most known-Control epochs
    - Decide “Control vs Task” either
         - with a P(Control) ≥ τ   (τ > 0)   **soft threshold**, or
         - by MAP hard assignment  (τ <= 0)  **cluster labels**
    - return aggregate metrics
    """
    gmm = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = seed).fit(X)

    # Posterior responsibilities: shape (n_epochs, 3)
    resp = gmm.predict_proba(X)

    # Identifying the component that best represents true "Control"
    ctrl_share = resp[ctrl_mask].mean(axis=0)          # mean P(ctrl) per comp
    ctrl_comp_id = int(np.argmax(ctrl_share))            # component index

    # Decision rule
    if tau <= 0.0:
        # MAP / hard assignment
        cluster_ids = gmm.predict(X)
        y_pred = np.where(cluster_ids == ctrl_comp_id, 0, 1)
    else:
        # Soft-probability threshold
        p_ctrl = resp[:, ctrl_comp_id]                  # P(Control) per epoch
        y_pred = np.where(p_ctrl >= tau, 0, 1)

    y_true = (~ctrl_mask).astype(int)                    # 0 = Control, 1 = Task

    #  Metrics
    beta     = 2.0
    tn, fp, fn, tp = confusion_matrix(
                        y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "f2"              : fbeta_score(y_true, y_pred, beta = beta, zero_division = 0),
        "precision_task"  : precision_score(y_true, y_pred, pos_label = 1, zero_division = 0),
        "recall_task"     : recall_score(y_true, y_pred, pos_label = 1, zero_division = 0),
        "f1_task"         : f1_score(y_true, y_pred, pos_label = 1, zero_division = 0),
        "accuracy"        : accuracy_score(y_true, y_pred),
        "fp_rate_control" : fp / (fp + tn + 1e-9),
        "ari"             : adjusted_rand_score(y_true, gmm.predict(X)),
        "bic"             : gmm.bic(X),
    }
    try:
        metrics["silhouette"] = silhouette_score(X, gmm.predict(X))
    except ValueError:
        metrics["silhouette"] = np.nan

    return metrics



#  MAIN search loop 
def main() -> None:
    # Building per-subject tables once
    print("\n=== Building per-subject feature tables ===")
    subj_tables = {s: _build_feature_table(s) for s in tqdm(SUBJECTS)}

    # Feature column tracking
    feat_cols = [c for c in subj_tables[SUBJECTS[0]].columns if c not in ("condition", "is_control")]

    ic_cols = [c for c in feat_cols if c.startswith("ic_")
                      or c.startswith("power_ic")]
    ic_power_cols = ["power_ic_left", "power_ic_right"]
    raw_power_cols = ["power_epoch"]
    raw_cols = [c for c in feat_cols if not c.startswith("ic_") and c not in ic_power_cols]

    candidate_sets: list[tuple[str, ...]] = [
        tuple(sorted(ic_cols)),        # 1
        tuple(sorted(ic_power_cols)),  # 2
        tuple(sorted(raw_power_cols)), # 3
        tuple(sorted(raw_cols)),       # 4
    ]

    for _ in range(N_RANDOM_TRIALS):
        k = RNG.randint(1, min(MAX_FEATURES, len(feat_cols)))
        candidate_sets.append(tuple(sorted(RNG.sample(feat_cols, k))))


    # EVALUATION

    print(f"\n=== Evaluating {len(candidate_sets)} subsets x {len(TAU_VALUES)} τ ===")
    seen, results, all_trials = set(), [], []
    progress_log, global_best_f2, iteration_idx = [], -1.0, 0

    for subset in tqdm(candidate_sets):
        if subset in seen:
            continue
        seen.add(subset)

        for tau in TAU_VALUES:
            iteration_idx += 1
            metrics_acc = defaultdict(list)
            for df in subj_tables.values():
                X = df.loc[:, subset].to_numpy(float)
                X = StandardScaler().fit_transform(np.nan_to_num(X, nan = 0.0))
                ctrl_mask = df["is_control"].to_numpy(bool)

                m = _fit_gmm_metrics(X, ctrl_mask, tau)
                for k, v in m.items():
                    metrics_acc[k].append(v)

            mean_m = {k: float(np.mean(vs)) for k, vs in metrics_acc.items()}
            mean_m |= {"tau" : tau, "subset" : ",".join(subset), "n_feat" : len(subset), "iter" : iteration_idx}
            all_trials.append(mean_m)

            # Tracking best tau per subset for later “winner” selection
            if (subset not in [r[0] for r in results] or mean_m["f2"] > [r[1]["f2"] for r in results if r[0]==subset][0]):
                # Removing old entry if exists
                results = [r for r in results if r[0] != subset]
                results.append((subset, mean_m))

            # Progress log whenever new global best is found
            if mean_m["f2"] > global_best_f2:
                global_best_f2 = mean_m["f2"]
                progress_log.append({
                    "iteration" : iteration_idx,
                    "subset"    : ",".join(subset),
                    "tau"       : tau,
                    "f2"        : mean_m["f2"],
                    "precision" : mean_m["precision_task"],
                    "recall"    : mean_m["recall_task"],
                    "fp_ctrl"   : mean_m["fp_rate_control"],
                })

    # Selecting overall best (max F2, tie --> fewer features)
    best_subset, best_m = max(results, key = lambda t: (t[1]["f2"], -t[1]["n_feat"]))

    # Save all trials
    out_dir = Path("search_results"); out_dir.mkdir(exist_ok = True)
    trials_path = out_dir / "random_search_trials.csv"
    pd.DataFrame(all_trials).to_csv(trials_path, index = False)

    prog_path = out_dir / "best_f2_progression.csv"
    pd.DataFrame(progress_log).to_csv(prog_path, index = False)

    # Summary for Terminal
    print("\n=== BEST FEATURE SET (by mean F₂) ===")
    print(f"Size {len(best_subset)}  -->  {best_subset}")
    print(f"τ used = {best_m['tau']}")
    for key in ("f2", "precision_task", "recall_task", "f1_task",
                "accuracy", "fp_rate_control", "ari", "bic", "silhouette"):
        print(f"{key:>18s}: {best_m[key]:.3f}")
    print(f"\nAll trials saved --> {trials_path.resolve()}")
    print(f"Best-so-far trajectory saved --> {prog_path.resolve()}")


if __name__ == "__main__":
    main()
