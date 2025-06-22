"""
Sanity checking by comparing proportions assigned to the “control” cluster (vs. all other clusters combined as “task”), with a tunable soft-decision threshold τ.

Scenario 1 (control only):
- Split Control: A (known) + B (held-out).
- Fit 3-comp GMM on A, pick control component.
- Fit 3-comp GMM on AuB, pick control componen.
- 2x2 table, chi-squared test H0: control/task (here more control) proportions equal.

Scenario 2 (control vs. task):
- Fit 3-comp GMM on all Control, pick control component.
- Fit 3-comp GMM on Control+Task, pick control component.
- 2x2 table, chi-squared test H0: control/task proportions equal.
"""

import re
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from finger_tapping.preprocessing import simple_pipeline, SUBJECTS
from finger_tapping.feature_preparation4 import extract_all_epoch_features, fit_motor_ica, DEFAULT_SAMPLING_RATE_HZ

SELECTED_FEATURES = ["power_ic_left", "power_ic_right"]

# TAU ≤ 0: hard MAP assignment
# TAU > 0: soft threshold on P(control) ≥ TAU
TAU = 0.95

RNG_SEED = 42

def normalise_conditions(raw):
    patt = {"Control": re.compile("control", re.I), "Left": re.compile("left", re.I), "Right": re.compile("right", re.I)}
    import mne
    ep = {name: raw[name] for name in raw.event_id}
    out = {}
    for std, p in patt.items():
        for k, v in ep.items():
            if p.search(k):
                out[std] = v
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
            fd = extract_all_epoch_features(arr, sampling_rate_hz = DEFAULT_SAMPLING_RATE_HZ, ica_model = ica, ic_left_index = ic_l, ic_right_index = ic_r)
            if not names:
                names = list(fd.keys())
            rows.append([fd[n] for n in names])
        feats[cond] = np.vstack(rows)
    # Scale selected features
    idxs = [names.index(f) for f in selected]
    sel  = {c: X[:,idxs] for c,X in feats.items()}
    allX = np.vstack(list(sel.values()))
    scaler = StandardScaler().fit(allX)
    return {c: scaler.transform(X) for c,X in sel.items()}

def pick_control_component(gmm, X_known):
    resp = gmm.predict_proba(X_known)
    return int(np.argmax(resp.mean(axis = 0)))

def label_control(gmm, X, ctrl_comp):
    if TAU <= 0.0:
        labels = gmm.predict(X)
        return labels == ctrl_comp
    else:
        resp = gmm.predict_proba(X)
        return resp[:, ctrl_comp] >= TAU

def scenario1_prop(feats):
    Xc = feats["Control"]
    A, B = train_test_split(Xc, test_size = 0.5, random_state = RNG_SEED)

    # Fit on A, label A
    g1 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = RNG_SEED).fit(A)
    ctrl1 = pick_control_component(g1, A)
    mask_A = label_control(g1, A, ctrl1)
    nA_ctrl = mask_A.sum()
    nA_task = len(A) - nA_ctrl

    # Fit on AuB, label AuB
    AB = np.vstack([A, B])
    g2 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = RNG_SEED).fit(AB)
    ctrl2 = pick_control_component(g2, A)
    mask_AB = label_control(g2, AB, ctrl2)
    nAB_ctrl = mask_AB.sum()
    nAB_task = len(AB) - nAB_ctrl

    table = np.array([[nA_ctrl,  nA_task], [nAB_ctrl, nAB_task]])
    _, p, _, _ = chi2_contingency(table)
    return table, p

def scenario2_prop(feats):
    Xc = feats["Control"]
    Xt = np.vstack([v for k,v in feats.items() if k!="Control"])

    # Fit on control, label control
    g1 = GaussianMixture(n_components = 3, covariance_type="full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state=RNG_SEED).fit(Xc)
    ctrl1 = pick_control_component(g1, Xc)
    mask_c = label_control(g1, Xc, ctrl1)
    nC_ctrl = mask_c.sum()
    nC_task = len(Xc) - nC_ctrl

    # Fit on control+task, label CT
    CT = np.vstack([Xc, Xt])
    g2 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = RNG_SEED).fit(CT)
    ctrl2 = pick_control_component(g2, Xc)
    mask_CT = label_control(g2, CT, ctrl2)
    nCT_ctrl = mask_CT.sum()
    nCT_task = len(CT) - nCT_ctrl

    table = np.array([[nC_ctrl,  nC_task], [nCT_ctrl, nCT_task]])
    _, p, _, _ = chi2_contingency(table)
    return table, p

def main():
    mode = "hard MAP" if TAU <= 0 else f"soft τ = {TAU}"
    print(f"\nProportion-based sanity check using {mode}\n")
    for subj in SUBJECTS:
        print(f"Subject {subj}:")
        raw = simple_pipeline(subj, save = False)
        edict = normalise_conditions(raw)
        feats = extract_features(edict, SELECTED_FEATURES)

        tbl1, p1 = scenario1_prop(feats)
        print("  Scenario 1 (Control split):")
        print("    Rows: [fit on A-->abel A,  fit on AuB-->label AuB]")
        print("    Cols: [assigned CONTROL, assigned TASK]")
        print(f"    Table = [[{tbl1[0,0]}, {tbl1[0,1]}],  # A-only (n={tbl1[0].sum()})")
        print(f"             [{tbl1[1,0]}, {tbl1[1,1]}]]  # AuB    (n={tbl1[1].sum()})")
        print(f"    chi-squared p = {p1:.4f}\n")

        tbl2, p2 = scenario2_prop(feats)
        print("  Scenario 2 (Control vs Task):")
        print("    Rows: [fit on CTRL-->label CTRL,  fit on CT-->label CT]")
        print("    Cols: [assigned CONTROL, assigned TASK]")
        print(f"    Table = [[{tbl2[0,0]}, {tbl2[0,1]}],  # CTRL only (n={tbl2[0].sum()})")
        print(f"             [{tbl2[1,0]}, {tbl2[1,1]}]]  # CTRL+Task (n={tbl2[1].sum()})")
        print(f"    chi-squared p = {p2:.4f}\n")

if __name__ == "__main__":
    main()
