"""
Chi-square test (or Fisher exact if very small counts) checking if:
- the cluster distribution is different between the Control and Tapping tasks,
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


def _prepare_table(df: pd.DataFrame) -> np.ndarray:
    """
    Build the 2 by 2 contingency table:
    rows  : Control, Tapping
    cols  : Cluster-1, Cluster-0
    """
    if "activity" not in df.index.names:
        raise ValueError("DataFrame index must contain the 'activity' level.")

    act  = df.index.get_level_values("activity").str.lower()
    task = np.where(act == "control", "Control", "Tapping")

    table = pd.crosstab(task, df["cluster"])
    table = (table.reindex(index=["Control", "Tapping"], columns=[1, 0])
                   .fillna(0).astype(int))
    return table.values


def cluster_task_chi2(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Always compute BOTH tests.
    Returns
    -------
    dict{
        table         : 2×2 ndarray
        chi2          : float
        p_chi2        : float
        fisher_or     : float   # odds-ratio
        p_fisher      : float
        cramer_v      : float   # np.nan if chi2 invalid
        valid_chi2    : bool    # False if any cell < 5
        significant   : bool    # using the *valid* test for alpha
    }
    """
    table = _prepare_table(df)
    a, b, c, d = table.ravel()
    n = a + b + c + d

    # Chi-square
    chi2, p_chi2, dof, _ = chi2_contingency(table, correction=False)
    cramer_v = np.sqrt(chi2 / n)

    # Validity checking for chi2 assumptions
    valid_chi2 = not (table < 5).any()

    if not valid_chi2:      # chi2 assumptions violated
        chi2, p_chi2, cramer_v = np.nan, np.nan, np.nan

    # Fisher exact
    fisher_or, p_fisher = fisher_exact(table)

    p_main = p_chi2 if valid_chi2 else p_fisher
    significant = p_main < alpha

    return {
        "table"     : table,
        "chi2"      : chi2,
        "p_chi2"    : p_chi2,
        "cramer_v"  : cramer_v,
        "valid_chi2": valid_chi2,
        "fisher_or" : fisher_or,
        "p_fisher"  : p_fisher,
        "significant": significant,
    }


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from finger_tapping.simple_GMM2 import simple_GMM

    subject = "01"

    _, df, _ = simple_GMM(subject, plot=False)   
    res = cluster_task_chi2(df)

    print("\nContingency table (rows: Control/Tapping, cols: Cluster-1/0):")
    print(res["table"])

    if res["valid_chi2"]:
        print(f"\nChi-square:  χ² = {res['chi2']:.2f},  p = {res['p_chi2']:.3g}, "
            f"Cramer's V = {res['cramer_v']:.3f}")
    else:
        print("\nChi-square assumptions violated (cell < 5)")

    print(f"Fisher exact:  odds-ratio = {res['fisher_or']:.3g}, "
        f"p = {res['p_fisher']:.3g}")

    print("\nSignificant at α = 0.05 ?",
        "YES" if res["significant"] else "NO",
        "(decision based on valid test)")
