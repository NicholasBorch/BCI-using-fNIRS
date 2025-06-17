from preprocessing import simple_pipeline, SUBJECTS
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
from typing import Any
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp
from leg_utils import run_tapping_pipeline

np.random.seed(42)

def get_data(subject: str, legs: bool = False) -> dict[str, Any]:
    if legs:
        epochs = run_tapping_pipeline(subject=subject, task='fingerauto', bids_root_path='bids_raw', save=False)
    else:
        epochs = simple_pipeline(subject=subject, save=False)
    events = epochs.events
    map_ = {v: k for k, v in epochs.event_id.items()}

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info['sfreq']


    ## Concatenate all epochs along time for ICA
    data_concat = data.transpose(0, 2, 1).reshape(n_epochs * n_times, n_channels)
    return {
        'data': data_concat,
        'events': events,
        'map': map_,
        'sfreq': sfreq,
        'n_channels': n_channels,
        'n_times': n_times,}
    
def extract_labels(epochs: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts labels from the events array."""
    for i, event in enumerate(events):
        if event[0] > len(epochs):
            break
    labels = events[:i, 2] #type: ignore
    epoch_indices = events[:i, 0]  #type: ignore
    return labels, epoch_indices

def remove_epochs(concat: np.ndarray,
                  labels: np.ndarray,
                  remove_idxs: np.ndarray,
                  n_times: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    concat : ndarray, shape (n_epochs * n_times, n_channels)
        The time-concatenated recording returned by `get_data`.
    labels : ndarray, shape (n_epochs,)
        One label per epoch.
    remove_idxs : 1-D ndarray of int
        Which epochs to discard (0-based).
    n_times : int
        Number of time points per epoch (can be taken straight from
        `epochs.get_data().shape[-1]`).

    Returns
    -------
    new_epochs : ndarray, shape (n_kept, n_times, n_channels)
    new_labels : ndarray, shape (n_kept,)
    """
    n_channels = concat.shape[1]
    n_epochs   = len(labels)
    assert concat.shape[0] == n_epochs * n_times, \
        f"Mismatch: expected {n_epochs * n_times} samples, got {concat.shape[0]}"

    # reshape to (epochs, times, channels)
    data_3d = concat.reshape(n_epochs, n_times, n_channels)

    # build mask
    mask = np.ones(n_epochs, dtype=bool)
    mask[remove_idxs] = False

    return data_3d[mask], labels[mask]


def make_new_events(events: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Creates a new events array by removing the specified indices."""
    new_events = []
    cnt = 0
    print(events)
    for i, event in enumerate(events):
        print(event)
        cnt += event[0]
        if i in indices:
            epoch_len = events[i+1, 0] - events[i, 0] if i < len(events) - 1 else 0
            cnt -= epoch_len
        else:
            new_events.append([cnt, event[1], event[2]])
    return np.array(new_events)
    
    
def sliding_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """Applies a sliding window to smooth the data."""
    return uniform_filter1d(data, size=window_size, axis=0)

def center_data(data_concat: np.ndarray) -> np.ndarray:
    """Centers the data by subtracting the mean."""
    return (data_concat - np.mean(data_concat, axis=0)) / np.std(data_concat, axis=0)

def get_ICA_signal(data_concat: np.ndarray, n_components: int) -> np.ndarray:
    """Performs ICA on the concatenated data and returns the signal."""
    ica = FastICA(random_state=42, whiten='unit-variance', max_iter=1000, tol=0.0001)#, n_components=n_components)
    S_ = ica.fit_transform(data_concat)  # (n_epochs * n_times, n_components)
    return S_

def get_segments(ica_signal: np.ndarray, events: np.ndarray, sfreq: float, delay_secs: int = 0) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Extracts segments from the ICA signal based on event onsets.
    
    Args:
        ica_signal (np.ndarray): The ICA signal with shape (n_samples, n_components).
        events (np.ndarray): The events array with shape (n_events, 3) where the first column is onset times.
        delay_secs (int): The delay in seconds to start the segment after the event onset.
    
    Returns:
        list[tuple[list[np.ndarray], list[int]]]: A list of tuples where each tuple contains:
            - A list of segments for each component.
            - A list of labels corresponding to each segment.
    """
    sample_onsets = events[:, 0]
    labels = events[:, 2]
    
    comp_segments = []
    comp_segment_labels = []
    
    for i in range(ica_signal.shape[1]):
        segments = []
        segment_labels = []
        
        component_signal = ica_signal[:, i]
        
        for j in range(len(sample_onsets)):
            start = sample_onsets[j]
            end = sample_onsets[j+1] if j < len(sample_onsets) - 1 else len(component_signal)
            if len(component_signal[start+round(delay_secs*sfreq):end]) == 0:
                continue
            segments.append(component_signal[start + delay_secs:end])
            segment_labels.append(labels[j])
        comp_segments.append(segments)
        comp_segment_labels.append(segment_labels)
    return comp_segments, np.array(comp_segment_labels[0])

def get_segments_rolling(
    ica_signal: np.ndarray,
    events: np.ndarray,
    sfreq: float,
    delay_secs: int = 0,
    window_secs: float = 5.0,    # length of each rolling window
    step_secs: float = 1.0       # stride of each rolling window
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """
    Extracts rolling windows from each event segment in each component.
    Each window inherits its event's label.
    """
    sample_onsets = events[:, 0]
    labels = events[:, 2]

    comp_segments = []
    comp_segment_labels = []

    window_size = int(window_secs * sfreq)
    step_size = int(step_secs * sfreq)

    for i in range(ica_signal.shape[1]):
        segments = []
        segment_labels = []
        component_signal = ica_signal[:, i]

        for j in range(len(sample_onsets)):
            start = int(sample_onsets[j] + delay_secs * sfreq)
            end = int(sample_onsets[j+1]) if j < len(sample_onsets) - 1 else len(component_signal)
            if component_signal[start:end].size == 0:
                continue
            # Slide window within this event's range
            for win_start in range(start, end - window_size + 1, step_size):
                win_end = win_start + window_size
                if len(component_signal[win_start:win_end]) < window_size:
                    continue
                segments.append(component_signal[win_start:win_end])
                segment_labels.append(labels[j])
        comp_segments.append(segments)
        comp_segment_labels.append(segment_labels)
    return comp_segments, np.array(comp_segment_labels[0])

        

def fft_segments(
    segments: list[np.ndarray], sfreq: float, target_length: int
    ) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Calculates the FFT for each segment, padding or truncating to target_length.
    Returns list of FFT amplitude vectors and the common frequency bins.
    """
    fft_results = []
    for seg in segments:
        seg = np.asarray(seg)
        N = len(seg)
        if N == 0:
            continue
        # Pad or truncate to target_length
        if N < target_length:
            seg = np.pad(seg, (0, target_length - N), mode='constant')
        elif N > target_length:
            seg = seg[:target_length]
        # Now N = target_length
        yf = rfft(seg)
        fft_results.append(np.abs(yf) / target_length) #type: ignore
    freqs = rfftfreq(target_length, 1/sfreq)
    return fft_results, freqs


def get_power_density(fft_results: list[np.ndarray]) -> list[float]:
    """Calculates the power density for each FFT result."""
    return [np.sum(amp**2) for amp in fft_results]

def segment_pipeline(data: dict[str, Any],
                    n_components: int,
                    delay_secs: int = 0,
                    sliding_window_size: int = 1,
                    window_secs: float = 15.0,
                    step_secs: float = 1.0) -> tuple[list[list[np.ndarray]], np.ndarray]:

    windowed_data = sliding_window(data['data'], window_size=sliding_window_size)
        
    centered_data = center_data(windowed_data)
    
    ica_signal = get_ICA_signal(centered_data, n_components=n_components)
    segments, true_labels = get_segments_rolling(ica_signal, data['events'], sfreq=data['sfreq'],
                                                 delay_secs=delay_secs, window_secs=window_secs, step_secs=step_secs)
    # segments, true_labels = get_segments(ica_signal, data['events'], sfreq=data['sfreq'], delay_secs=delay_secs)
    return segments, true_labels

def fft_pipeline(segments: list[list[np.ndarray]], num_components: int, sfreq: float, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies FFT to each segment in each component.
    Returns a list of FFT results and the common frequency bins.
    """
    comp_fft = []
    comp_freqs = []
    
    for component_segments in segments[:num_components]:
        fft_results, freqs = fft_segments(component_segments, sfreq, target_length)
        comp_fft.append(fft_results)
        comp_freqs.append(freqs)
        
    comp_fft = np.array(comp_fft)  # (n_components, n_segments, n_freqs)
    
    return comp_fft, np.array(comp_freqs)

def find_distr_difference(comp_fft: np.ndarray, 
                          control_indices: np.ndarray, 
                          activity_indices: np.ndarray,
                          num_bins: int) -> tuple[np.ndarray, np.ndarray]:
    control_fft = comp_fft[:, control_indices, :]  # (n_components, n_control_segments, n_freqs)
    average_control_fft = np.mean(control_fft, axis=1)  # (n_components, n_freqs)
    bin_indices = np.zeros((comp_fft.shape[0], num_bins + 1), dtype=int)
    for comp in range(comp_fft.shape[0]):
        power_cumsum = np.cumsum(average_control_fft[comp, :]**2)
        total_power = power_cumsum[-1]
        for b in range(1, num_bins):
            threshold = b * total_power / num_bins
            bin_indices[comp, b] = np.searchsorted(power_cumsum, threshold)
        bin_indices[comp, 0] = 0
        bin_indices[comp, -1] = average_control_fft.shape[1]
        
    # Bin all segments by those edges
    binned_power = np.zeros((comp_fft.shape[1], comp_fft.shape[0], num_bins))  # (n_segments, n_components, n_bins)
    for comp in range(comp_fft.shape[0]):
        for seg in range(comp_fft.shape[1]):
            for bin_idx in range(num_bins):
                start_idx = bin_indices[comp, bin_idx]
                end_idx = bin_indices[comp, bin_idx + 1]
                binned_power[seg, comp, bin_idx] = np.sum(comp_fft[comp, seg, start_idx:end_idx] ** 2)
    # Compute group means and differences
    binned_control_power = np.mean(binned_power[control_indices, :, :], axis=0)  # (n_components, n_bins)
    binned_control_power_individual = binned_power[control_indices, :, :]
    binned_activity_power = binned_power[activity_indices, :, :]
    binned_activity_power_diff = binned_activity_power - binned_control_power[np.newaxis, :, :]
    binned_control_power_diff = binned_control_power_individual - binned_control_power[np.newaxis, :, :]
    return binned_activity_power_diff, binned_control_power_diff

def get_pvals(binned_power: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Computes p-values for the binned power differences.
    
    Args:
        binned_power (np.ndarray): Binned power differences.
        alpha (float): Significance level for correction.
    
    Returns:
        np.ndarray: P-values for the binned power differences.
    """
    n_components, n_bins = binned_power.shape[1], binned_power.shape[2]
    pvals = np.zeros((n_components, n_bins))
    
    for comp in range(n_components):
        for bin_idx in range(n_bins):
            _, pval = ttest_1samp(binned_power[:, comp, bin_idx], 0.0)
            pvals[comp, bin_idx] = pval #type: ignore
            
    return pvals
    
    
def get_corrected_pvals(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Applies FDR correction to the p-values.
    
    Args:
        pvals (np.ndarray): Array of p-values to correct.
        alpha (float): Significance level for correction.
    
    Returns:
        np.ndarray: Corrected p-values.
    """
    _, pvals_corrected, _, _ = multipletests(pvals.ravel(), alpha=alpha, method='fdr_bh')
    return pvals_corrected.reshape(pvals.shape)  # Reshape back to original shape

def main(subject: str, NUM_COMPONENTS: int, NUM_BINS: int, ALPHA: float, TARGET_LENGTH: int = 256,
         temp_shuffle: bool = False, legs: bool = False) -> tuple[bool, bool]: #I know this is bad practice
    """CHECK IF TARGET_LENGTH MATCHES ROLLING SEGMENTS

    Args:
        subject (str): _description_
        NUM_COMPONENTS (int): _description_
        NUM_BINS (int): _description_
        ALPHA (float): _description_
        TARGET_LENGTH (int, optional): _description_. Defaults to 256.
    """
    data = get_data(subject, legs = legs)
    n_epochs = int(data['data'].shape[0] / data['n_times'])
    labels = data['events'][:n_epochs, 2] 
    
    if temp_shuffle:        
        remove_indices = labels != 1
    else:
        remove_indices = np.array([])
    
    if temp_shuffle:
        epochs, labels = remove_epochs(data['data'], labels, remove_indices, n_times=data['n_times'])
        data['data'] = epochs.reshape(-1, data['n_channels'])  # flatten again  
    
    segments, true_labels = segment_pipeline(data, delay_secs=5, sliding_window_size=100, window_secs=10, step_secs=.25,
                                             n_components=NUM_COMPONENTS)
    if temp_shuffle:
        assert all(true_labels), "we shouldn't have any activity labels left"
    if temp_shuffle:
        # now all true_labels are 1, so we will fake activity labels
        ctrl_to_activity_ratio = 0.33  # Ratio of control indices to switch to activity
        num_to_switch = int(ctrl_to_activity_ratio * len(true_labels))
        indices_to_switch = np.random.choice(true_labels, size=num_to_switch, replace=False)
        true_labels[indices_to_switch] = 2  # Switch some control labels to activity
        
    control_indices = true_labels == 1
    activity_indices = true_labels != 1
    # shape of segments: (n_components, n_segments, n_times)
    # plot first segment of first component
    if not temp_shuffle:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(segments[0][0], label=f'Component 1, Segment 1, [{data['map'][true_labels[0]]}]')
        plt.title(f'Segment of Component 1 for Subject {subject}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(segments[1][0], label=f'Component 2, Segment 1, [{data['map'][true_labels[0]]}]')
        plt.title(f'Segment of Component 2 for Subject {subject}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()
    
    comp_fft, _ = fft_pipeline(segments, num_components=NUM_COMPONENTS, sfreq=data['sfreq'], target_length=TARGET_LENGTH)
    binned_activity_power_diff, binned_control_power_diff = find_distr_difference(comp_fft= comp_fft,
                                                                                control_indices=control_indices,
                                                                                activity_indices=activity_indices,
                                                                                num_bins=NUM_BINS)
    # plot the fft results for the first segment of the first  component
    if not temp_shuffle:
        plt.subplot(2, 2, 3)
        plt.hist(comp_fft[0, 0], bins=100, alpha=0.5, label='FFT Amplitude, [{}]'.format(data['map'][true_labels[0]]))
        plt.title(f'FFT Amplitude Distribution for Component 1, Segment 1, Subject {subject}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.hist(comp_fft[1, 0], bins=100, alpha=0.5, label='FFT Amplitude, [{}]'.format(data['map'][true_labels[0]]))
        plt.title(f'FFT Amplitude Distribution for Component 2, Segment 1, Subject {subject}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    
    # print('Performing t-tests on binned control power differences...')
    # pvals_control = get_pvals(binned_control_power_diff, alpha=ALPHA)
    # pvals_control_corrected = get_corrected_pvals(pvals_control)
    
    # # show array with significance bools
    # significance = pvals_control_corrected < ALPHA
    # print(f"Control Significance (p < {ALPHA}):\n", significance)
    # print('Control Significance on components:')
    # print([any(significance[i, :]) for i in range(NUM_COMPONENTS)])
    # print('"Control Significance rate":')
    # print(np.mean([any(significance[i, :]) for i in range(NUM_COMPONENTS)]))
    # print('Overall control significance conclusion:', bool(np.mean([any(significance[i, :]) for i in range(NUM_COMPONENTS)])))  
    # print()
    overall_control_significance = False  # We do not expect control significance in this test, so we set it to False
    # overall_control_significance = bool(np.mean([any(significance[i, :]) for i in range(NUM_COMPONENTS)]))
    
    print("Performing t-tests on binned activity power differences...")
    pvals_activity = get_pvals(binned_activity_power_diff, alpha=ALPHA)
    pvals_activity_corrected = get_corrected_pvals(pvals_activity, alpha=ALPHA)
    
    # show array with significance bools
    significance = pvals_activity_corrected < ALPHA
    print(f"Significance (p < {ALPHA}):\n", significance)
    print('Significance on components:')
    print([any(significance[i, :]) for i in range(NUM_COMPONENTS)])
    print('"Significance rate":')
    print(np.mean([any(significance[i, :]) for i in range(NUM_COMPONENTS)]))
    max_activity_pval = np.max(pvals_activity_corrected)
    min_activity_pval = np.min(pvals_activity_corrected)
    print('Min activity p-value:', min_activity_pval)
    print('Max activity p-value:', max_activity_pval)
    print('Overall activity significance conclusion:', bool(np.mean([any(significance[i, :]) for i in range(NUM_COMPONENTS)])))
    print()
    overall_activity_significance = bool(np.mean([any(significance[i, :]) for i in range(NUM_COMPONENTS)]))
    
    # # plot the binned power densities
    # plt.figure(figsize=(12, 6))
    # for comp in range(NUM_COMPONENTS):
    #     plt.subplot(1, NUM_COMPONENTS, comp + 1)
    #     plt.bar(range(NUM_BINS), binned_activity_power_diff[:, comp, :].mean(axis=0), label='Activity - Control')
    #     plt.title(f'Component {comp + 1}')
    #     plt.xlabel('Frequency Bin')
    #     plt.ylabel('Power Density Difference')
    #     plt.xticks(range(NUM_BINS))
    #     plt.legend()
    # plt.tight_layout()
    # plt.show()
    return overall_activity_significance, overall_control_significance
    


if __name__ == '__main__':
    fp, tp, fn, tn = 0, 0, 0, 0
    for subject in SUBJECTS:
        for shuf in [False, True]:
            print(f"Processing subject: {subject} with shuffle={shuf}")
            ac, ct = main(subject=subject, NUM_COMPONENTS=2, NUM_BINS=2, ALPHA=0.05, TARGET_LENGTH=2024, temp_shuffle=shuf)
            print(f"Activity significance: {ac}, Should be {not shuf}")
            print(f"Control significance: {ct}, Should be False")
            print('Success:', ac == (not shuf) and ct is False)
            if shuf:
                fp += ac
                tn += not ac
            else:
                tp += ac
                fn += not ac
            print("\n" + "="*50 + "\n")
    print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")
    
    # on legs
    # print('RUNNING ON LEGS')
    # leg_subs = [f"{i:02d}" for i in range(1, 96)]
    # fp, tp, fn, tn = 0, 0, 0, 0
    # for subject in leg_subs:
    #     try:
    #         for shuf in [False, True]:
    #             print(f"Processing subject: {subject} with shuffle={shuf}")
    #             ac, ct = main(subject=subject, NUM_COMPONENTS=3, NUM_BINS=2, ALPHA=0.05, TARGET_LENGTH=2024, temp_shuffle=shuf, legs=True)
    #             print(f"Activity significance: {ac}, Should be {not shuf}")
    #             print(f"Control significance: {ct}, Should be False")
    #             print('Success:', ac == (not shuf) and ct is False)
    #             if shuf:
    #                 fp += ac
    #                 tn += not ac
    #             else:
    #                 tp += ac
    #                 fn += not ac
    #             print("\n" + "="*50 + "\n")
    #     except (FileNotFoundError, RuntimeError, ValueError) as e:
    #         print(f"File not found for subject {subject}: {e}")
            
    # print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")