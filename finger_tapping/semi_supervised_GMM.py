"""
This is a Semi-supervised Gaussian-Mixture pipeline that classifies each fNIRS epoch as
“control” (0) or “task” (1).

Features per epoch ( **HbO only** )
---------------------------------------------------------------------------------------------------------
1. Δ-mean:                          |       Mean(activation) - mean(baseline)
2. Δ-variance                       |       Var(activation) - var(baseline)
3. linear-slope                     |       Ordinary-least-squares slope through activation part*
4. **AUC-diff**                     |       Area-under-curve(activation) - area-under-curve(baseline)
5. Δ band-power                     |       0.01-0.08 Hz
6. Δ band-power                     |       0.08-0.15 Hz
7. Δ band-power                     |       0.15-0.50 Hz

* Note that "Slope" is the single straight-line fit through the whole activation window.

Weighting / semi-supervision
---------------------------------------------------------------------------------------------------------
* All **control** epochs are duplicated *CONTROL_SAMPLE_WEIGHT* times so the
  GMM is encouraged to dedicate one cluster to “rest”.
* After clustering we map the cluster with the highest share of known control
  epochs to label 0 and finally force every known control epoch to 0.

Three evaluation modes
---------------------------------------------------------------------------------------------------------
1. Task-cued (real):                        True task vs control from the experiment  
2. Synthetic:                               Half the control epochs are faked as “task”  
3. Synthetic (Injected):                    Same as (2) + one real task epoch injected
"""

# Library imports
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import mne

from preprocessing import simple_pipeline, SUBJECTS

# Constants
SAMPLING_RATE_HZ = 7.81 
BASELINE_DURATION_SEC = 5.0
FREQUENCY_BANDS = [(0.01, 0.08), (0.08, 0.15), (0.15, 0.50)]
CONTROL_SAMPLE_WEIGHT = 3.0
RANDOM_SEED = 42

# Utility functions
def baseline_samples_per_epoch(n_timepoints: int) -> int:
    """Return number of baseline samples before the cue."""
    return int(round(BASELINE_DURATION_SEC * SAMPLING_RATE_HZ))


def band_power(signal: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Welch band-power for the specified band."""
    freqs, power = welch(signal, fs = SAMPLING_RATE_HZ, nperseg = min(64, len(signal)))
    return power[(freqs >= f_lo) & (freqs < f_hi)].sum()


def extract_epoch_features(epoch_hbo: np.ndarray) -> np.ndarray:
    """
    Parameters:
    epoch_hbo: (n_channels, n_timepoints) HbO epoch

    Returns:
    1-D numpy array with 7 features (see module docstring).
    """
    n_channels, n_timepoints = epoch_hbo.shape
    n_baseline = baseline_samples_per_epoch(n_timepoints)

    # Channel-average trace
    avg_trace = epoch_hbo.mean(axis=0)
    baseline_signal = avg_trace[:n_baseline]
    activation_signal = avg_trace[n_baseline:]

    # Delta-mean and Delta-variance
    delta_mean = activation_signal.mean() - baseline_signal.mean()
    delta_variance = activation_signal.var() - baseline_signal.var()

    # Activation window slope
    time_axis = np.arange(len(activation_signal)) / SAMPLING_RATE_HZ
    slope, *_ = linregress(time_axis, activation_signal)

    # AUC - baseline corrected
    auc_activation = np.trapz(activation_signal, dx=1 / SAMPLING_RATE_HZ)
    auc_baseline   = np.trapz(baseline_signal,   dx=1 / SAMPLING_RATE_HZ)
    auc_difference = auc_activation - auc_baseline

    # Delta-band-powers
    delta_powers = []
    for low_f, high_f in FREQUENCY_BANDS:
        p_act  = band_power(activation_signal, low_f, high_f)
        p_base = band_power(baseline_signal,   low_f, high_f)
        delta_powers.append(p_act - p_base)

    return np.array([delta_mean, delta_variance, slope, auc_difference, *delta_powers], dtype=float)


def extract_features_from_epochs(epochs: mne.Epochs, label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of Epochs into a feature matrix X and label vector y.
    Each row in X corresponds to one epoch.
    """
    feature_rows = [extract_epoch_features(epochs[i].get_data(picks="hbo")[0]) for i in range(len(epochs))]
    labels = np.full(len(feature_rows), label, dtype=int)
    return np.vstack(feature_rows), labels


def synthetic_split(control_epochs: mne.Epochs, frac_task: float = 0.50) -> tuple[mne.Epochs, mne.Epochs]:
    """Randomly split control epochs into pseudo-task and true-control sets."""
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(control_epochs))
    cut = int(round(frac_task * len(idx)))
    return control_epochs[idx[:cut]], control_epochs[idx[cut:]]


def synthetic_split_and_inject(control_epochs: mne.Epochs, one_task_epoch: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    Half of the control epochs become fake-task. We inject one real task epoch.
    Returns feature matrix X and true label vector y.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(len(control_epochs))
    half = len(shuffled) // 2

    true_ctrl = control_epochs[shuffled[:half]]
    fake_task = control_epochs[shuffled[half:]]

    X_ctrl, y_ctrl   = extract_features_from_epochs(true_ctrl, 0)
    X_fake, y_fake   = extract_features_from_epochs(fake_task, 1)
    X_real, y_real   = extract_features_from_epochs(one_task_epoch[:1], 1)

    X = np.vstack([X_ctrl, X_fake, X_real])
    y = np.hstack([y_ctrl, y_fake, y_real])
    return X, y

# Defining pipeline
def run_task_detection_pipeline(subject_id: str, *, synthetic: bool = False, injected: bool = False) -> None:
    """Run the semi-supervised GMM pipeline for one subject in a chosen mode."""
    mode = ("Synthetic (Injected)" if injected else "Synthetic" if synthetic else "Task-cued (real)")
    print(f"\n{'='*7} {mode} {'='*7}")

    # 1) Load the preprocessed epochs for the subject
    epochs = simple_pipeline(subject_id, save=True)
    n_ctrl, n_left, n_right = map(len, (epochs["Control"], epochs["Left"], epochs["Right"]))
    print(f"Epochs: Control={n_ctrl}, Left={n_left}, Right={n_right}")

    # 2) Build feature matrix and true labels for the chosen scenario
    if injected:
        real_task_epoch = epochs["Left"][:1] if n_left else epochs["Right"][:1]
        X_all, y_true = synthetic_split_and_inject(epochs["Control"], real_task_epoch)

    elif synthetic:
        pseudo_task, true_ctrl = synthetic_split(epochs["Control"])
        X_ctrl, y_ctrl = extract_features_from_epochs(true_ctrl, 0)
        X_task, y_task = extract_features_from_epochs(pseudo_task, 1)
        X_all = np.vstack([X_ctrl, X_task])
        y_true = np.hstack([y_ctrl, y_task])

    else:
        left, right = epochs["Left"], epochs["Right"]
        task_epochs = left.copy()
        task_epochs._data = np.concatenate([left.get_data(), right.get_data()], axis=0)
        task_epochs.events = np.vstack([left.events, right.events])
        task_epochs.selection = np.concatenate([left.selection, right.selection])
        task_epochs.drop_log = left.drop_log + right.drop_log

        X_ctrl, y_ctrl = extract_features_from_epochs(epochs["Control"], 0)
        X_task, y_task = extract_features_from_epochs(task_epochs, 1)
        X_all = np.vstack([X_ctrl, X_task])
        y_true = np.hstack([y_ctrl, y_task])

    # 3) Standard-scale the feature matrix
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)

    # 4) Weight control epochs by duplication
    n_control = sum(y_true == 0)
    control_dup_features = np.repeat(X_scaled[:n_control], int(CONTROL_SAMPLE_WEIGHT), axis=0)
    control_dup_labels = np.repeat(y_true[:n_control], int(CONTROL_SAMPLE_WEIGHT), axis=0)

    X_weighted = np.vstack([control_dup_features, X_scaled[n_control:]])
    y_weighted = np.hstack([control_dup_labels,  y_true[n_control:]])

    # 5) Train GMM
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=RANDOM_SEED, n_init=20, reg_covar=1e-6)
    gmm.fit(X_weighted)
    cluster_assignment = gmm.predict(X_scaled)

    # 6) Map clusters to control/task labels
    control_cluster = (pd.DataFrame({'cluster': cluster_assignment, 'is_ctrl': (y_true == 0).astype(int)}).groupby('cluster')['is_ctrl'].mean().idxmax())
    y_pred = np.where(cluster_assignment == control_cluster, 0, 1)
    y_pred[:n_control] = 0 # Force known controls to label 0 (task-cued will be 1 after this step)

    # 7) Print evaluation results
    print(classification_report(y_true, y_pred, digits=2))
    print("\n=== Cluster composition (ctrl/task) ===")
    print(pd.DataFrame({'lbl': y_true, 'c': cluster_assignment}).groupby('c')['lbl'].value_counts().unstack(fill_value=0), "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    first_subject = SUBJECTS[0]
    run_task_detection_pipeline(first_subject, synthetic=False)
    run_task_detection_pipeline(first_subject, synthetic=True)
    run_task_detection_pipeline(first_subject, injected=True)
