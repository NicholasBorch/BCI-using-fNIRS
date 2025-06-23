"""
Resampling-based Clinical Sanity Check
- Scenario 1: Monte Carlo splits on Control (100 reps).

In each rep we:
  1) Fit GMM on control-only --> pick “control” component --> count NCa, NTa
  2) Fit GMM on enlarged set --> re-identify same component --> count NCb, NTb
  3) Table = [[NCa,NTa],[NCb,NTb]], chi-squared p-value
Then summarize mean proportions, 95% CIs, and false-positive/sensitivity rates.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from finger_tapping.preprocessing import simple_pipeline, SUBJECTS
from finger_tapping.feature_preparation4 import extract_all_epoch_features, fit_motor_ica, DEFAULT_SAMPLING_RATE_HZ

# CHOOSE FEATURE CONFIGURATION
# SELECTED_FEATURES = ["delta_mean", "delta_variance"]
# SELECTED_FEATURES = ["power_ic_left", "power_ic_right"]
SELECTED_FEATURES = ['abs_auc_activation', 'ic_left_delta_variance', 'ic_left_extrema_line_length', 'power_ic_right']
TAU       = 0           # ≤0: hard MAP; >0: P(control) ≥ TAU
RNG_SEED  = 42
N_MC      = 100         # Monte Carlo splits for Scenario 1
TASKS     = ["Left","Right"]
K         = 1 + len(TASKS)
# ──────────────────────────────────────────────────────────────────────────────

def normalise_conditions(raw):
    patt = {"Control": re.compile("control", re.I), "Left": re.compile("left", re.I), "Right": re.compile("right", re.I)}
    ep = {n: raw[n] for n in raw.event_id}
    out = {}
    for lbl, pat in patt.items():
        for k,v in ep.items():
            if pat.search(k):
                out[lbl] = v
                break
    if "Control" not in out:
        raise RuntimeError("Control condition missing")
    return out

def extract_features(ep_dict, selected):
    ica, ic_l, ic_r = fit_motor_ica(ep_dict)
    feats, names = {}, []
    for cond, epochs in ep_dict.items():
        data = epochs.get_data(picks=["hbo","hbr"])
        rows = []
        for arr in data:
            fd = extract_all_epoch_features(arr, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, ica_model=ica, ic_left_index=ic_l, ic_right_index=ic_r)
            if not names:
                names = list(fd.keys())
            rows.append([fd[n] for n in names])
        feats[cond] = np.vstack(rows)
    # Select and scale
    idxs = [names.index(f) for f in selected]
    sel  = {c: X[:,idxs] for c,X in feats.items()}
    allX = np.vstack(list(sel.values()))
    scaler = StandardScaler().fit(allX)
    return {c: scaler.transform(X) for c,X in sel.items()}

def pick_control_component(gmm, X_known):
    post = gmm.predict_proba(X_known)
    return int(np.argmax(post.mean(axis=0)))

def label_control(gmm, X, ctrl_comp):
    if TAU <= 0:
        return (gmm.predict(X) == ctrl_comp)
    post = gmm.predict_proba(X)
    return (post[:,ctrl_comp] >= TAU)

def scenario1_resample(feats):
    Xc = feats["Control"]
    rng = np.random.RandomState(RNG_SEED)
    pA_list, pAB_list, pvals = [], [], []
    for _ in range(N_MC):
        seed = rng.randint(0,2**31-1)
        A,B = train_test_split(Xc, test_size=0.5, random_state=seed)
        AB  = np.vstack([A,B])

        # Fit on A
        g1 = GaussianMixture(n_components=K, covariance_type="full", n_init=20, max_iter=500, reg_covar=1e-6, random_state=seed).fit(A)
        c1   = pick_control_component(g1, A)
        mA   = label_control(g1, A,   c1)
        NCa, NTa = mA.sum(), len(A)-mA.sum()

        # Fit on A∪B
        g2 = GaussianMixture(n_components=K, covariance_type="full", n_init=20, max_iter=500, reg_covar=1e-6, random_state=seed).fit(AB)
        c2   = pick_control_component(g2, A)
        mAB  = label_control(g2, AB, c2)
        mB   = mAB[len(A):]
        NCb, NTb = mB.sum(), len(B)-mB.sum()

        table = np.array([[NCa,NTa],[NCb,NTb]])
        _, p,_,_ = chi2_contingency(table)

        pA_list.append(NCa/(NCa+NTa))
        pAB_list.append(NCb/(NCb+NTb))
        pvals.append(p)

    # Summarize
    meanA, meanAB = np.mean(pA_list), np.mean(pAB_list)
    ciA   = np.percentile(pA_list,  [2.5,97.5])
    ciAB  = np.percentile(pAB_list, [2.5,97.5])
    fp_rate = np.mean(np.array(pvals)<0.05)*100
    return meanA, ciA, meanAB, ciAB, fp_rate

def main():
    print(f"\n=== Clinical Sanity Check (hard MAP, K={K}) ===\n")
    for subj in SUBJECTS:
        print(f"Subject {subj}:")
        raw   = simple_pipeline(subj, save=False)
        edict = normalise_conditions(raw)
        feats = extract_features(edict, SELECTED_FEATURES)

        mA,ciA,mAB,ciAB,fp = scenario1_resample(feats)
        print(f"  Scenario 1 (MC splits, N={N_MC}):")
        print(f"    Control-only frac = {mA:.3f} (95% CI [{ciA[0]:.3f},{ciA[1]:.3f}])")
        print(f"    AuB          frac = {mAB:.3f} (95% CI [{ciAB[0]:.3f},{ciAB[1]:.3f}])")
        print(f"    False-positive rate (p<0.05): {fp:.1f}%\n")

if __name__=="__main__":
    main()
