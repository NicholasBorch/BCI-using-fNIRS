#!/usr/bin/env python3
"""
Extended clinical sanity check with:
 1) Monte Carlo splits for Scenario 1 --> bootstrap CIs + record “bad” splits + save all splits.
 2) Bootstrap for Scenario 2 --> bootstrap CIs + record “misses” + save all bootstraps.
 3) Single-split group-level pooling across subjects (≈ Cochran-Mantel-Haenszel).

Saves:
 - sanity_subject_results.csv
 - sanity_group_pooled.csv
 - sanity_scenario1_splits.csv
 - sanity_scenario2_bootstraps.csv
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


SELECTED_FEATURES   = ["power_ic_left", "power_ic_right"]
TAU                 = 0.95       # ≤0: hard MAP; >0: soft threshold
N_REPEATS           = 100        # Monte Carlo repeats for Scenario 1
N_BOOTSTRAP         = 100        # Bootstrap reps for Scenario 2
CI_LOWER, CI_UPPER  = 2.5, 97.5  # Percentiles for 95% CI


RNG_SEED = 42

def normalise_conditions(raw):
    patt = {"Control": re.compile("control", re.I), "Left": re.compile("left", re.I), "Right": re.compile("right", re.I)}
    import mne
    ep = {n: raw[n] for n in raw.event_id}
    out = {}
    for label, pat in patt.items():
        for k, v in ep.items():
            if pat.search(k):
                out[label] = v
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
    idxs = [names.index(f) for f in selected]
    sel = {c: X[:,idxs] for c,X in feats.items()}
    allX = np.vstack(list(sel.values()))
    scaler = StandardScaler().fit(allX)
    return {c: scaler.transform(X) for c,X in sel.items()}

def pick_control_component(gmm, X_known):
    resp = gmm.predict_proba(X_known)
    return int(np.argmax(resp.mean(axis = 0)))

def label_control(gmm, X, ctrl_comp):
    if TAU <= 0.0:
        return (gmm.predict(X) == ctrl_comp)
    resp = gmm.predict_proba(X)
    return (resp[:,ctrl_comp] >= TAU)

def scenario1_repeat(feats, seed):
    """
    Monte Carlo repeat for Scenario 1. Returns (seed, nA, nB, pA, pAB, chi-squared pval)
    """
    Xc = feats["Control"]
    A, B = train_test_split(Xc, test_size = 0.5, random_state=seed)
    nA, nB = len(A), len(B)

    g1 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = seed).fit(A)
    ctrl1  = pick_control_component(g1, A)
    mask_A = label_control(g1, A, ctrl1)

    AB = np.vstack([A, B])
    g2 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = seed+1).fit(AB)
    ctrl2   = pick_control_component(g2, A)
    mask_AB = label_control(g2, AB, ctrl2)

    table = np.array([[mask_A.sum(), nA - mask_A.sum()], [mask_AB.sum(), (nA+nB) - mask_AB.sum()]])
    _, pval, _, _ = chi2_contingency(table)

    return {"seed": seed, "nA": nA, "nB": nB, "pA": mask_A.mean(), "pAB": mask_AB.mean(), "pval": pval}

def scenario1_counts(feats, seed):
    """Return the 2x2 pooled count table for Scenario 1 (fixed seed)."""
    Xc = feats["Control"]
    A, B = train_test_split(Xc, test_size = 0.5, random_state=seed)

    g1 = GaussianMixture(n_components = 3, covariance_type = "full",
        n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = seed).fit(A)
    ctrl1 = pick_control_component(g1, A)
    maskA = label_control(g1, A, ctrl1)

    AB = np.vstack([A,B])
    g2 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state=seed+1).fit(AB)
    ctrl2  = pick_control_component(g2, A)
    maskAB = label_control(g2, AB, ctrl2)

    return np.array([[maskA.sum(), len(A) - maskA.sum()], [maskAB.sum(), len(A)+len(B) - maskAB.sum()]])

def scenario2_bootstrap(feats, seed):
    """
    Bootstrap replicate for Scenario 2. Returns dict with seed, counts and p-values.
    """
    rng = np.random.RandomState(seed)
    Xc = feats["Control"]
    Xt = np.vstack([v for k,v in feats.items() if k!="Control"])

    idx_c = rng.randint(0, len(Xc), size=len(Xc))
    idx_t = rng.randint(0, len(Xt), size=len(Xt))
    Xc_b, Xt_b = Xc[idx_c], Xt[idx_t]

    g1 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = seed).fit(Xc_b)
    ctrl1 = pick_control_component(g1, Xc_b)
    mask_c = label_control(g1, Xc_b, ctrl1)
    nC_ctrl = mask_c.sum()
    nC_task = len(Xc_b) - nC_ctrl

    CT_b = np.vstack([Xc_b, Xt_b])
    g2 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state=seed+1).fit(CT_b)
    ctrl2 = pick_control_component(g2, Xc_b)
    mask_CT = label_control(g2, CT_b, ctrl2)
    nCT_ctrl = mask_CT.sum()
    nCT_task = len(CT_b) - nCT_ctrl

    table = np.array([[nC_ctrl, nC_task], [nCT_ctrl, nCT_task]])
    _, pval, _, _ = chi2_contingency(table)

    return {"seed": seed, "nC_ctrl": nC_ctrl, "nC_task": nC_task, "nCT_ctrl": nCT_ctrl, "nCT_task": nCT_task, "pC": mask_c.mean(), "pCT": mask_CT.mean(), "pval": pval}

def scenario2_once(feats):
    """Single-split Scenario 2 --> returns 2x2 table + p-value."""
    Xc = feats["Control"]
    Xt = np.vstack([v for k,v in feats.items() if k!="Control"])

    g1 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state=RNG_SEED).fit(Xc)
    ctrl1 = pick_control_component(g1, Xc)
    mask_c = label_control(g1, Xc, ctrl1)

    CT = np.vstack([Xc, Xt])
    g2 = GaussianMixture(n_components = 3, covariance_type = "full", n_init = 20, max_iter = 500, reg_covar = 1e-6, random_state = RNG_SEED).fit(CT)
    ctrl2 = pick_control_component(g2, Xc)
    mask_CT = label_control(g2, CT, ctrl2)

    table = np.array([[mask_c.sum(), len(Xc) - mask_c.sum()], [mask_CT.sum(), len(CT) - mask_CT.sum()]])
    _, pval, _, _ = chi2_contingency(table)
    return table, pval

def main():
    rng = np.random.RandomState(RNG_SEED)

    # CSV
    subject_rows   = []
    splits_rows1   = []  # All scenario1 repeats
    boots_rows2    = []  # All scenario2 bootstraps
    pooled1, pooled2 = np.zeros((2,2),int), np.zeros((2,2),int)

    print(f"\nScenario 1 (Monte Carlo splits, N={N_REPEATS}):\n")
    for subj in SUBJECTS:
        raw   = simple_pipeline(subj, save = False)
        edict = normalise_conditions(raw)
        feats = extract_features(edict, SELECTED_FEATURES)

        # Scenario 1 repeats
        results1, bad_info = [], []
        for _ in range(N_REPEATS):
            info = scenario1_repeat(feats, rng.randint(0,2**31-1))
            info["subject"] = subj
            results1.append(info)
            splits_rows1.append(info)
            if info["pval"] <= 0.05:
                bad_info.append((info["seed"], info["nA"], info["nB"]))

        df1 = pd.DataFrame(results1)
        mA, mAB = df1["pA"].mean(),  df1["pAB"].mean()
        ciA = np.percentile(df1["pA"],  [CI_LOWER, CI_UPPER])
        ciAB = np.percentile(df1["pAB"], [CI_LOWER, CI_UPPER])
        passFrac1 = (df1["pval"] > 0.05).mean()*100

        print(f"Subject {subj}:")
        print(f"  A-only frac = {mA:.3f} (95% CI [{ciA[0]:.3f},{ciA[1]:.3f}])")
        print(f"  A∪B   frac = {mAB:.3f} (95% CI [{ciAB[0]:.3f},{ciAB[1]:.3f}])")
        print(f"  χ² p>0.05 in {passFrac1:.1f}% of splits")
        if bad_info:
            print("  Rejected splits (seed,|A|,|B|):")
            for seed,nA,nB in bad_info[:10]:
                print(f"    seed={seed}, A={nA}, B={nB}")
            if len(bad_info)>10:
                print("    ...")
        print()

        pooled1 += scenario1_counts(feats, RNG_SEED)

        # Scenario 2 bootstrap
        print(f"Scenario 2 (bootstrap N={N_BOOTSTRAP}):")
        results2, miss_info = [], []
        for _ in range(N_BOOTSTRAP):
            rec = scenario2_bootstrap(feats, rng.randint(0,2**31-1))
            rec["subject"] = subj
            results2.append(rec)
            boots_rows2.append(rec)
            if rec["pval"] > 0.05:
                miss_info.append((rec["seed"], rec["nC_ctrl"], rec["nC_task"], rec["nCT_ctrl"], rec["nCT_task"]))

        df2 = pd.DataFrame(results2)
        mC, mCT = df2["pC"].mean(), df2["pCT"].mean()
        ciC = np.percentile(df2["pC"],  [CI_LOWER, CI_UPPER])
        ciCT = np.percentile(df2["pCT"], [CI_LOWER, CI_UPPER])
        sensFrac2 = (df2["pval"] < 0.05).mean()*100

        print(f"  CTRL-only frac = {mC:.3f} (95% CI [{ciC[0]:.3f},{ciC[1]:.3f}])")
        print(f"  CTRL+T   frac = {mCT:.3f} (95% CI [{ciCT[0]:.3f},{ciCT[1]:.3f}])")
        print(f"  χ² p<0.05 in {sensFrac2:.1f}% of bootstraps")
        if miss_info:
            print("  Missed bootstraps (seed, C-only/T-only/C+T-only counts):")
            for seed,nc,nt,nct,tt in miss_info[:10]:
                print(f"    seed={seed}, C-only({nc}/{nt}), C+T({nct}/{tt})")
            if len(miss_info)>10:
                print("    ...")
        print()

        tbl2, p2 = scenario2_once(feats)
        pooled2 += tbl2
        print(f"Scenario 2 single-split: table={tbl2.tolist()}, χ² p={p2:.4f}\n")

        # Record summary row
        subject_rows.append({
            "subject":         subj,
            "s1_mean_pA":      mA,   "s1_CIlo_pA":  ciA[0],  "s1_CIhi_pA":  ciA[1],
            "s1_mean_pAB":     mAB,  "s1_CIlo_pAB": ciAB[0], "s1_CIhi_pAB": ciAB[1],
            "s1_pass_pct":     passFrac1,
            "s1_bad_splits":   ";".join(f"{s}:{a}|{b}" for s,a,b in bad_info),
            "s2_mean_pC":      mC,   "s2_CIlo_pC":  ciC[0],  "s2_CIhi_pC":  ciC[1],
            "s2_mean_pCT":     mCT,  "s2_CIlo_pCT": ciCT[0], "s2_CIhi_pCT": ciCT[1],
            "s2_sens_pct":     sensFrac2,
            "s2_missed":       ";".join(f"{s}:{nc}/{nt}/{nct}/{tt}" for s,nc,nt,nct,tt in miss_info),
            "s2_once_C_ctrl":  int(tbl2[0,0]), "s2_once_C_task":  int(tbl2[0,1]),
            "s2_once_CT_ctrl": int(tbl2[1,0]), "s2_once_CT_task": int(tbl2[1,1]),
            "s2_once_p":       p2
        })

    # 1) Save subject-level summary
    df_subj = pd.DataFrame(subject_rows)
    df_subj.to_csv("sanity_subject_results.csv", index = False)

    # 2) Save all scenario1 splits
    df_s1 = pd.DataFrame(splits_rows1)
    df_s1.to_csv("sanity_scenario1_splits.csv", index = False)

    # 3) Save all scenario2 bootstraps
    df_s2 = pd.DataFrame(boots_rows2)
    df_s2.to_csv("sanity_scenario2_bootstraps.csv", index = False)

    # 4) Save pooled group tables
    pooled_rows = []
    for scenario,table in [("Scenario1", pooled1), ("Scenario2", pooled2)]:
        rows = [("first","row1"),("second","row2")]
        for i,(rname,_) in enumerate(rows):
            pooled_rows.append({"scenario": scenario, "row_name": rname, "in_control": int(table[i,0]), "in_task": int(table[i,1])})
    df_pool = pd.DataFrame(pooled_rows)
    df_pool.to_csv("sanity_group_pooled.csv", index=False)

    # 5) Print pooled chi-squared
    print("Group-level pooled chi-squared results:")
    _, p1g, _, _ = chi2_contingency(pooled1)
    _, p2g, _, _ = chi2_contingency(pooled2)
    print(f"  Scenario 1 pooled χ² p = {p1g:.4f}")
    print(f"  Scenario 2 pooled χ² p = {p2g:.4f}")

    print("\n--> Saved subject summary --> sanity_subject_results.csv")
    print("--> Saved splits data    --> sanity_scenario1_splits.csv")
    print("--> Saved bootstrap data --> sanity_scenario2_bootstraps.csv")
    print("--> Saved pooled tables  --> sanity_group_pooled.csv")

if __name__ == "__main__":
    main()
