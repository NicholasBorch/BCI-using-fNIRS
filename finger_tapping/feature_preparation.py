"""
Preparing fNIRS features from pre-processed epochs
--------------------------------------------------------------------------
This module extracts a set of features from a single epoch of fNIRS data
"""

from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.stats  import kurtosis
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score

# CONSTANTS
DEFAULT_SAMPLING_RATE_HZ:   float = 7.81
BASELINE_DURATION_SEC:      float = 5.0
WELCH_NPERSEG_MAX:            int = 64

# Broadband limits for the energy feature
FREQ_LOW_HZ:  float = 0.01
FREQ_HIGH_HZ: float = 1.20

# Helper functions
def _baseline_len(n_samples: int, fs: float) -> int:
    """
    Calculates the number of samples corresponding to the baseline period.

    Parameters:
    n_samples : int
        Total number of samples in the epoch.
    fs : float
        Sampling frequency in Hz.

    Returns:
    int
        Number of samples in the 5-second baseline.
    """
    return int(round(BASELINE_DURATION_SEC * fs))

def _features_from_trace(trace: NDArray[np.float32], fs: float = DEFAULT_SAMPLING_RATE_HZ, prefix: str = "", include_power: bool = True) -> Dict[str, float]:
    """
    Extracts 9 descriptive features from a 1D fNIRS signal trace.

    Parameters:
    trace : np.ndarray
        1D array representing the time series.
    fs : float
        Sampling frequency in Hz.
    prefix : str
        Prefix for feature keys (used for IC features).
    include_power : bool
        Whether to include power_epoch in the output (used for raw only).

    Returns:
    dict
        Dictionary of extracted features.
    """
    n_bl = _baseline_len(trace.size, fs)
    baseline, activation = trace[:n_bl], trace[n_bl:]

    # Amplitude and dispersion features
    delta_mean         = abs(activation).mean() - abs(baseline).mean()
    peak_amplitude     = activation.max() - baseline.mean()
    delta_variance     = activation.var(ddof=0) - baseline.var(ddof=0)
    auc_activation     = np.trapz(activation, dx=1 / fs)
    auc_baseline       = np.trapz(baseline, dx=1 / fs)
    auc_difference     = auc_activation - auc_baseline
    abs_auc_activation = abs(auc_activation)
    if include_power:
        power_epoch = float(np.mean(trace ** 2))

    # Shape descriptors
    idx_peak, idx_trough = int(np.argmax(activation)), int(np.argmin(activation))
    if idx_peak != idx_trough:
        extrema_abs_slope  = abs((activation[idx_peak] - activation[idx_trough]) / (idx_peak - idx_trough))
        extrema_line_length = np.hypot(idx_peak - idx_trough, activation[idx_peak] - activation[idx_trough])
    else:
        extrema_abs_slope = extrema_line_length = 0.0
    kurtosis_activation = kurtosis(activation, bias=False)

    feats = {
        "delta_mean"            : delta_mean,
        "peak_amplitude"        : peak_amplitude,
        "delta_variance"        : delta_variance,
        "auc_difference"        : auc_difference,
        "abs_auc_activation"    : abs_auc_activation,
        "kurtosis_activation"   : kurtosis_activation,
        "extrema_abs_slope"     : extrema_abs_slope,
        "extrema_line_length"   : extrema_line_length,
    }
    if include_power:
        feats["power_epoch"] = power_epoch

    return {f"{prefix}{k}": float(v) for k, v in feats.items()}

def fit_motor_ica(epochs_by_condition: Dict[str, "mne.Epochs"], random_state: int = 42) -> Tuple[FastICA, int, int]:
    """
    Fits ICA on all epochs and selects motor-relevant components.

    Parameters:
    epochs_by_condition : dict
        Dictionary mapping condition labels ("Control", "Left", "Right") to MNE Epochs.
    random_state : int
        Seed for ICA reproducibility.

    Returns:
    tuple
        - Trained FastICA model
        - Index of left motor IC
        - Index of right motor IC
    """
    cond_order = ["Control", "Left", "Right"]
    blocks, n_samp = [], []

    for cond in cond_order:
        dat = epochs_by_condition[cond].get_data(picks=["hbo", "hbr"])
        resh = dat.transpose(0, 2, 1).reshape(-1, dat.shape[1])
        blocks.append(resh)
        n_samp.append(resh.shape[0])

    X = np.vstack(blocks)
    ica = FastICA(n_components=X.shape[1], whiten="unit-variance", max_iter=800, tol=1e-3, random_state=random_state)
    S = ica.fit_transform(X)

    # Estimate power per component, per epoch
    rows, labs, cur = [], [], 0
    for i, cond in enumerate(cond_order):
        n_ep = epochs_by_condition[cond].get_data().shape[0]
        seg = S[cur:cur + n_samp[i]]
        cur += n_samp[i]
        rows.append(seg.reshape(n_ep, -1, S.shape[1]).var(axis=1))
        labs.extend([i] * n_ep)

    P = np.vstack(rows)
    labs = np.array(labs)

    # Discriminate between Left and Right
    mask = np.isin(labs, [1, 2])
    X_lr = P[mask]
    y_lr = (labs[mask] == 2).astype(int)

    auc = np.array([roc_auc_score(y_lr, X_lr[:, j]) for j in range(X_lr.shape[1])])
    rank = np.argsort(np.abs(auc - 0.5))[::-1]

    ic_right = int(rank[0])
    ic_left  = int(rank[1])
    return ica, ic_left, ic_right

def extract_all_epoch_features(
    epoch_data: NDArray[np.float32],
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    ica_model: Optional[FastICA] = None,
    ic_left_index: Optional[int] = None,
    ic_right_index: Optional[int] = None
) -> Dict[str, float]:
    """
    Extracts all features from a single fNIRS epoch, optionally using ICA.

    Parameters:
    epoch_data : np.ndarray
        Shape (channels, timepoints), fNIRS data from a single epoch.
    sampling_rate_hz : float
        Sampling frequency in Hz.
    ica_model : FastICA, optional
        Pre-fitted ICA model.
    ic_left_index : int, optional
        Index of left motor IC.
    ic_right_index : int, optional
        Index of right motor IC.

    Returns:
    dict
        Dictionary of extracted features (10 to 30 features depending on ICA).
    """
    feats = _features_from_trace(epoch_data.mean(axis=0), fs=sampling_rate_hz, include_power=True)

    if ica_model is None:
        return feats

    if ic_left_index is None or ic_right_index is None:
        raise ValueError("IC indices required when ICA model is supplied")

    src = ica_model.transform(epoch_data.T)
    ic_l, ic_r = src[:, ic_left_index], src[:, ic_right_index]

    feats.update({
        "power_ic_left" : float(np.mean(ic_l ** 2)),
        "power_ic_right": float(np.mean(ic_r ** 2)),
        **_features_from_trace(ic_l, fs=sampling_rate_hz, prefix="ic_left_", include_power=False),
        **_features_from_trace(ic_r, fs=sampling_rate_hz, prefix="ic_right_", include_power=False),
    })
    return feats


###
if __name__ == "__main__":
    from finger_tapping.preprocessing import simple_pipeline, SUBJECTS

    subj = SUBJECTS[0]
    epd  = simple_pipeline(subj, save=False)

    ica, ic_l, ic_r = fit_motor_ica(epd)
    first = epd["Control"][0].get_data(picks = ["hbo", "hbr"])[0]
    fdict = extract_all_epoch_features(first, ica_model = ica, ic_left_index = ic_l, ic_right_index = ic_r)

    print(f"{len(fdict)} features extracted for subject {subj}")
    for k, v in fdict.items():
        print(f"{k:28s}: {v: .6f}")
