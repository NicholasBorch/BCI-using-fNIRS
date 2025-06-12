from finger_tapping.feature_preparation2 import extract_X_y
from finger_tapping.preprocessing import simple_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
    
def plot_pca_2D(df, pc_x: int, pc_y: int, figsize=(8, 6), plot = True, hue='label'):
    """
    Plotting, pca index starts from 0 e.g plot pca1 and pca2 then set pc_x=0, pc_y=
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=pc_x, y=pc_y, hue=hue, alpha=0.7, palette="Set1")
    plt.xlabel(f'Principal Component {pc_x + 1}')
    plt.ylabel(f'Principal Component {pc_y + 1}')
    plt.title('PCA on Finger Tapping Data')
    plt.legend(title="Class")
    plt.grid()
    plt.show()
    
def plot_ica_2D(df, ic_x: int, ic_y: int, figsize=(8, 6), hue='label'):
    """
    Plotting, pca index starts from 0 e.g plot pca1 and pca2 then set pc_x=0, pc_y=1
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=ic_x, y=ic_y, hue=hue, alpha=0.7, palette="Set1")
    plt.xlabel(f'Independent Component {ic_x + 1}')
    plt.ylabel(f'Independent Component {ic_y + 1}')
    plt.title('ICA on Finger Tapping Data')
    plt.legend(title="Class")
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_pca_weights(pca_components, figsize=(10, 4)):
    """
    Plot PCA component weights
    """
    plt.figure(figsize=figsize)
    plt.bar(range(pca_components.components_.shape[1]), pca_components.components_[0])
    plt.title('Sensor Contributions to PC1 (Motor Laterality?)')
    plt.xlabel('Channels')
    plt.ylabel('Weight')
    plt.show()
    
def plot_ica_weights(ica_componets, figsize=(10, 4)):
    """
    Plot ICA component weights
    """
    plt.figure(figsize=figsize)
    plt.bar(range(ica_componets.mixing_.shape[0]), ica_componets.mixing_[:, 0])
    plt.title('Sensor Contributions to IC1 (Motor Laterality?)')
    plt.xlabel('Channels')
    plt.ylabel('Weight')
    plt.show()

def plot_ica_timecourses(df, figsize=(12, 3), sort_chronologically=True):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        The ICA dataframe returned by `run_ica()` (columns = IC1…ICn, index =
        MultiIndex['activity', 'epoch', 't_in_epoch']).
    figsize : tuple
        Size per subplot. The final figure height scales with n_components.
    sort_chronologically : bool
        If True, rows are sorted by ('epoch', 't_in_epoch') before plotting,
        so time runs left→right regardless of how the dataframe is shuffled.
    """
    if sort_chronologically and isinstance(df.index, pd.MultiIndex):
        df_plot = df.sort_index(level=["epoch", "t_in_epoch"])
    else:
        df_plot = df.copy()

    n_components = df_plot.shape[1] - 1 if "label" in df_plot else df_plot.shape[1]

    fig, axes = plt.subplots(
        n_components, 1, sharex=True,
        figsize=(figsize[0], figsize[1] * n_components),
        constrained_layout=True)

    if n_components == 1:   # when n_components=1 axes is not a list
        axes = [axes]

    # Plot each IC
    for k, ax in enumerate(axes):
        ic_name = df_plot.columns[k]
        ax.plot(df_plot[ic_name].values, lw=.8)
        ax.set_ylabel(ic_name)
        ax.spines[['right', 'top']].set_visible(False)

    axes[-1].set_xlabel("Samples (chronological)")
    fig.suptitle("Independent Components over Time", y=1.02, fontsize=14)
    plt.show()


def run_pca(subject, n_components = 5, random_state = 42, include_hbr=False):
    """
    Running PCA on subject data
    """
    X, y, idx = extract_X_y(subject, include_hbr=include_hbr, return_index=True)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Combine in dataframe
    pca_df = pd.DataFrame(X_pca)
    pca_df['label'] = y
    return pca, pca_df
    

def run_ica(subject, n_components = 5, random_state = 42, include_hbr=False):
    """
    Running ICA on subject data
    """
    X, y = extract_X_y(subject, include_hbr=include_hbr, return_index=False)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply ICA
    ica = FastICA(n_components = n_components, max_iter=1000, tol=0.0001, random_state = random_state)
    X_ica = ica.fit_transform(X_scaled)

    # Prepare DataFrame
    ica_df = pd.DataFrame(X_ica)
    ica_df['label'] = y
    return ica, ica_df


def plot_ica_pairwise(df: pd.DataFrame, hue: str = "label", corner: bool = True, diag_kind: str = "kde", figsize: tuple = (12, 12), markers: list | None = None, palette: str | dict | None = None):
    """
    Pair-wise scatter matrix of all ICA components with class labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by ``run_ica`` (i.e. IC1…ICn plus *hue* column).
    hue : str, default "label"
        Name of the column that contains the class labels
        (e.g. ``'control'``, ``'left'``, ``'right'``).
    corner : bool, default True
        If True, only the lower-triangle is drawn (cleaner for many ICs).
    diag_kind : {'hist', 'kde', 'auto'}, default 'kde'
        Type of plot shown on the diagonal.
    figsize : tuple, default (12, 12)
        Base figure size; height scales automatically with component count.
    markers : list, optional
        Matplotlib marker styles for the classes. Falls back to
        ``['o', 's', '^']`` if not given.
    palette : str | dict, optional
        Colour palette.  If ``None`` the Seaborn ``"Set1"`` palette is used.

    Returns
    -------
    seaborn.axisgrid.PairGrid
        The grid object for further tweaking (e.g. ``g.savefig('foo.png')``).
    """
    # Identify IC columns (everything except *hue*)
    comp_cols = [c for c in df.columns if c != hue]

    # Ensure a categorical type so legend order stays Control–Left–Right
    if hue in df.columns and df[hue].dtype == object:
        df = df.copy()  # avoid mutating caller
        df[hue] = pd.Categorical(df[hue], categories=["control", "left", "right"], ordered=True)

    g = sns.pairplot(df, vars=comp_cols, hue=hue, corner=corner, diag_kind=diag_kind, markers=markers or ['o', 's', '^'], palette=palette or "Set1", plot_kws=dict(alpha=0.65, linewidth=0))
    # Make the figure a bit roomier
    g.fig.set_size_inches(figsize)
    g.fig.suptitle("Pairwise ICA Component Scatter Matrix", y=1.02, fontsize=14)
    plt.show()
    return g


if __name__ == '__main__':
    
    # Load data
    subject = simple_pipeline(subject="01")

    # Run PCA
    pca, pca_df= run_pca(subject, include_hbr=False)

    # Run ICA
    ica, ica_df = run_ica(subject, include_hbr=False)
    
    # plot pca
    plot_pca_2D(pca_df, pc_x=0, pc_y=1)
    plot_ica_2D(ica_df, ic_x=0, ic_y=1)
    plot_pca_weights(pca)
    plot_ica_weights(ica)

    plot_ica_timecourses(ica_df)
    plot_ica_timecourses(ica_df, sort_chronologically=False)

    # After obtaining ica_df
    plot_ica_pairwise(ica_df)                 # Full matrix, lower triangle only
    plot_ica_pairwise(ica_df, corner=False)   # Full upper & lower triangle
