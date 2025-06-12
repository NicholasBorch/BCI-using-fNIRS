import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

from finger_tapping.preprocessing import simple_pipeline
from finger_tapping.pca_ica2_0    import run_ica


def simple_GMM(subject_id: str, n_components_ica: int = 5, sampling_rate_hz: float = 7.81, random_state: int = 42, include_hbr: bool = False, plot: bool = True, return_index: bool = True) -> tuple:
    """
    Fits a 2-cluster Gaussian-Mixture Model to all ICA dimensions for one subject and plots every IC against time, coloured by cluster label. Red dashed lines marks Control, Left, Right transitions.
    """
    # Preprocesssing and ICA
    epochs = simple_pipeline(subject=subject_id)
    ica, df = run_ica(epochs, n_components=n_components_ica, random_state=random_state, include_hbr=include_hbr, return_index=return_index)

    # Renaming columns to IC1, IC2, ..., if not already named
    if not any(isinstance(c, str) and c.startswith("IC") for c in df.columns):
        df = df.rename(columns={col: f"IC{idx+1}" for idx, col in enumerate(df.columns) if col != "label"})
    df = df.drop(columns=["label"], errors="ignore")

    ic_cols = [c for c in df.columns if c.startswith("IC")]
    df["time"] = np.arange(len(df)) / sampling_rate_hz

    # Gaussian Mixture Model on All ICs
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    df["cluster"] = gmm.fit_predict(df[ic_cols])

    # Visualisation
    if plot:
        # Detecting task-switch boundaries
        if isinstance(df.index, pd.MultiIndex) and "activity" in df.index.names:
            act_ser   = pd.Series(df.index.get_level_values("activity"))
            change_ix = np.flatnonzero(act_ser.ne(act_ser.shift()))[1:]
            vlines    = df.iloc[change_ix]["time"].values 
        else:
            seg_len = len(df) // 3
            vlines  = df.iloc[[seg_len, 2 * seg_len]]["time"].values

        n_ics = len(ic_cols)
        fig, axes = plt.subplots(n_ics, 1, sharex=True, figsize=(12, 2.7 * n_ics), constrained_layout=True)

        if n_ics == 1:
            axes = [axes]

        for ic, ax in zip(ic_cols, axes):
            sns.scatterplot(ax=ax, data=df, x="time", y=ic, hue="cluster", palette="Set2", s=8, legend=False)
            for x in vlines:
                ax.axvline(x=x, color="red", ls="--", lw=1)
            ax.set_ylabel(ic)
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Subject {subject_id} – GMM (k=3) on {n_ics} ICs",
                     y=1.02, fontsize=14)
        plt.show()

    return ica, df, gmm


if __name__ == "__main__":
    simple_GMM("01", plot=True)
