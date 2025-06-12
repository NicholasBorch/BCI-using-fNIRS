"""
Checking if method works if only fed with control epochs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt             
import seaborn as sns                       
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture

from finger_tapping.preprocessing import simple_pipeline
from finger_tapping.feature_preparation import _select_channels
from finger_tapping.eval_GMM2 import cluster_task_chi2


def control_only_test( subject_id: str, n_components_ica: int = 5, sampling_rate_hz: float = 7.81, include_hbr: bool = False, random_state: int = 42, plot: bool = True):
    epochs = simple_pipeline(subject=subject_id)["Control"].copy()
    epochs = _select_channels(epochs, include_hbr=include_hbr)

    # Reshape
    data = epochs.get_data()              
    n_ep, n_ch, n_t = data.shape
    X = data.swapaxes(1, 2).reshape(n_ep * n_t, n_ch)
    X_std = StandardScaler().fit_transform(X)

    # ICA
    ica = FastICA(n_components=n_components_ica, random_state=random_state, max_iter=1000, tol=1e-4)
    X_ic = ica.fit_transform(X_std)
    ic_cols = [f"IC{i+1}" for i in range(X_ic.shape[1])]
    df = pd.DataFrame(X_ic, columns=ic_cols)

    # GMM
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    df["cluster"] = gmm.fit_predict(df[ic_cols])

    # Fake task split
    half = len(df) // 2
    activity = np.array(["control"] * half + ["tapping"] * (len(df) - half))
    epoch_idx = np.repeat(np.arange(n_ep), n_t)[: len(df)]
    t_idx = np.tile(np.arange(n_t), n_ep)[: len(df)]
    df.index = pd.MultiIndex.from_arrays(
        [activity, epoch_idx, t_idx],
        names=["activity", "epoch", "t_in_epoch"],
    )
    df["time"] = np.arange(len(df)) / sampling_rate_hz

    # 6) Chi-square / Fisher
    stats = cluster_task_chi2(df)

    print(f"\n── Subject {subject_id}  (control-only sanity check) ──")
    print("Contingency table (rows Control/Tapping, cols Cluster-1/0):")
    print(stats["table"])

    if stats["valid_chi2"]:
        print(f"\nChi-square:  χ² = {stats['chi2']:.2f}, " f"p_chi2 = {stats['p_chi2']:.3g}, " f"Cramer's V = {stats['cramer_v']:.3f}")
    else:
        print("\nChi-square:  assumptions violated (cell < 5)")

    print(f"Fisher exact:  odds-ratio = {stats['fisher_or']:.3g}, " f"p_fisher = {stats['p_fisher']:.3g}")

    print("\nSignificant at alpha = 0.05 ?", "YES" if stats["significant"] else "NO", "(decision based on valid test)")

    
    # Plot ICs against time, coloured by cluster label
    if plot:
        n_ics = len(ic_cols)
        fig, axes = plt.subplots(n_ics, 1, sharex=True, figsize=(12, 2.7 * n_ics), constrained_layout=True)

        if n_ics == 1:
            axes = [axes]

        vline_x = df.iloc[half]["time"]        

        for ic, ax in zip(ic_cols, axes):
            sns.scatterplot(ax=ax, data=df, x="time", y=ic, hue="cluster", palette="Set2", s=8, legend=False)
            ax.axvline(x=vline_x, color="red", ls="--", lw=1)
            ax.set_ylabel(ic)
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Subject {subject_id} - control-only ICs (GMM k = 2)", y=1.02, fontsize=14)
        plt.show()



# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    control_only_test(subject_id="01", n_components_ica=5, sampling_rate_hz=7.81, random_state=42, plot=True)
