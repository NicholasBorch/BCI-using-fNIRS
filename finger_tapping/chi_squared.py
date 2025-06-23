"""
Proportion-based sanity check exactly as:

- Scenario 1 (control-only splits): split Control into A/B.
    - Fit GMM on A --> identify “control” cluster --> count NCa, NTa --> top row [NCa, NTa]
    - Fit GMM on AuB --> re-identify control cluster --> count on B only NCb, NTb --> bottom row [NTb, NCb]
- Scenario 2 (control vs task):
    - Fit GMM on Control --> count NCc, NTc --> top row [NCc, NTc]
    - Fit GMM on Control+Task --> re-identify control cluster --> count on Task only NCt, NTt --> bottom row [NTt, NCt]

Here K = 1 + len(SCENARIO2_EPOCHS).
"""
import re
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from finger_tapping.preprocessing import simple_pipeline, SUBJECTS
from finger_tapping.feature_preparation import extract_all_epoch_features, fit_motor_ica, DEFAULT_SAMPLING_RATE_HZ

# FEATURE CONFIGURATION
# SELECTED_FEATURES = ["delta_mean", "delta_variance"]
# SELECTED_FEATURES = ["power_ic_left", "power_ic_right"]
SELECTED_FEATURES = ['abs_auc_activation', 'ic_left_delta_variance', 'ic_left_extrema_line_length', 'power_ic_right']
TAU               = 0       # ≤0: hard MAP; >0: P(control) ≥ TAU
RNG_SEED          = 42
SCENARIO2_EPOCHS  = ["Left", "Right"]  # choose which task epochs to include



def normalise_conditions(raw):
    patt = {"Control": re.compile("control", re.I), "Left": re.compile("left", re.I), "Right": re.compile("right", re.I)}
    ep = {name: raw[name] for name in raw.event_id}
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
    resp = gmm.predict_proba(X_known)
    return int(np.argmax(resp.mean(axis=0)))

def label_control(gmm, X, ctrl_comp):
    if TAU <= 0:
        return (gmm.predict(X) == ctrl_comp)
    resp = gmm.predict_proba(X)
    return (resp[:, ctrl_comp] >= TAU)

def scenario1_prop(feats, K):
    Xc = feats["Control"]
    A, B = train_test_split(Xc, test_size=0.5, random_state=RNG_SEED)
    AB  = np.vstack([A, B])

    # Fit on A only
    g1 = GaussianMixture(n_components=K, covariance_type="full",n_init=20, max_iter=500,reg_covar=1e-6, random_state=RNG_SEED).fit(A)
    ctrl1 = pick_control_component(g1, A)
    maskA = label_control(g1, A, ctrl1)
    NCa, NTa = maskA.sum(), len(A)-maskA.sum()

    # Fit on A∪B
    g2 = GaussianMixture(n_components=K, covariance_type="full", n_init=20, max_iter=500, reg_covar=1e-6, random_state=RNG_SEED).fit(AB)
    ctrl2 = pick_control_component(g2, A)
    maskAB = label_control(g2, AB, ctrl2)
    maskB  = maskAB[len(A):]        
    NCb, NTb = maskB.sum(), len(B)-maskB.sum()

    table = np.array([[NCa, NTa],
                       [NCb, NTb]])
    _, pval, _, _ = chi2_contingency(table)
    return table, pval

def scenario2_prop(feats, tasks, K):
    Xc = feats["Control"]
    # Fit on Control only
    g1 = GaussianMixture(n_components=K, covariance_type="full", n_init=20, max_iter=500, reg_covar=1e-6, random_state=RNG_SEED).fit(Xc)
    ctrl1 = pick_control_component(g1, Xc)
    maskC = label_control(g1, Xc, ctrl1)
    NCc, NTc = maskC.sum(), len(Xc)-maskC.sum()

    # Fit on Control+Task
    X_t = np.vstack([feats[t] for t in tasks]) if tasks else np.empty((0, Xc.shape[1]))
    CT  = np.vstack([Xc, X_t])
    g2 = GaussianMixture(n_components=K, covariance_type="full", n_init=20, max_iter=500, reg_covar=1e-6, random_state=RNG_SEED).fit(CT)
    ctrl2 = pick_control_component(g2, Xc) 
    maskCT = label_control(g2, CT, ctrl2)
    maskT  = maskCT[len(Xc):]      
    NCt, NTt = maskT.sum(), len(maskT)-maskT.sum()

    table = np.array([[NCc, NTc],
                       [NCt, NTt]])
    _, pval, _, _ = chi2_contingency(table)
    return table, pval

def main():
    K    = 1 + len(SCENARIO2_EPOCHS)
    mode = "hard MAP" if TAU<=0 else f"soft τ={TAU}"
    print(f"\n=== Proportion-based sanity check using {mode}, K={K} ===\n")

    for subj in SUBJECTS:
        print(f"Subject {subj}:")
        raw   = simple_pipeline(subj, save=False)
        edict = normalise_conditions(raw)
        feats = extract_features(edict, SELECTED_FEATURES)

        tbl1, p1 = scenario1_prop(feats, K)
        print("  Scenario 1 (control-only split):")
        print(f"    Table = {tbl1.tolist()}   χ² p = {p1:.4f}")

        tbl2, p2 = scenario2_prop(feats, SCENARIO2_EPOCHS, K)
        print(f"  Scenario 2 (control + {SCENARIO2_EPOCHS}):")
        print(f"    Table = {tbl2.tolist()}   χ² p = {p2:.4f}\n")

if __name__=="__main__":
    main()
