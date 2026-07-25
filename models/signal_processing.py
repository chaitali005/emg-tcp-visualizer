"""
Signal processing functions for EMG-style multichannel data.

This module is intentionally framework-free: no PySide6, no sockets, no
plotting. It only operates on plain NumPy arrays of shape
(channels, samples), so it can be reused identically for:

- the live rolling buffer coming from TcpClientModel
- the full offline recording coming from RecordingBuffer

Processing parameters (documented here so the README can reference them):

- Bandpass filter: 4th-order Butterworth, 20-450 Hz
  (typical EMG passband; removes movement artifact / DC drift below 20 Hz
  and high-frequency noise above 450 Hz, well under the 1000 Hz Nyquist
  frequency for a 2000 Hz sampling rate).
- RMS window: 100 ms, centered (symmetric) window around each sample.
"""

import numpy as np
from scipy import signal


def apply_bandpass_filter(
    data: np.ndarray,
    sampling_rate: float,
    low_cut: float = 20.0,
    high_cut: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to each channel.

    Parameters
    ----------
    data : np.ndarray
        Shape (channels, samples). Works for a single channel too,
        as long as it is passed in as shape (1, samples).
    sampling_rate : float
        Sampling rate in Hz.
    low_cut : float
        Lower cutoff frequency in Hz. Must be > 0.
    high_cut : float
        Upper cutoff frequency in Hz. Must be < Nyquist (sampling_rate / 2).
    order : int
        Filter order (before zero-phase doubling via filtfilt).

    Returns
    -------
    np.ndarray
        Filtered data, same shape and dtype as the input.

    Notes
    -----
    `scipy.signal.filtfilt` requires a minimum number of samples relative
    to the filter order (roughly 3 * (2 * order + 1)). If there is not yet
    enough data -- which happens during the first moments of a live
    stream, before the rolling buffer has filled up -- this function
    returns the input unchanged rather than raising, so callers (e.g. a
    ViewModel polling on every timer tick) do not need to special-case
    this themselves.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (channels, samples), got shape {data.shape}")

    nyquist = sampling_rate / 2

    if low_cut <= 0:
        raise ValueError("low_cut must be greater than 0 Hz.")
    if high_cut >= nyquist:
        raise ValueError(
            f"high_cut ({high_cut} Hz) must be below the Nyquist frequency "
            f"({nyquist} Hz) for a sampling rate of {sampling_rate} Hz."
        )
    if low_cut >= high_cut:
        raise ValueError("low_cut must be smaller than high_cut.")

    num_samples = data.shape[1]
    min_required_samples = 3 * (2 * order + 1)

    if num_samples < min_required_samples:
        # Not enough data yet for a stable filtfilt result (e.g. right
        # after connecting, before the rolling buffer has filled up).
        # Returning the unfiltered data avoids crashing the caller and
        # is visually indistinguishable for such a short window anyway.
        return data.copy()

    low = low_cut / nyquist
    high = high_cut / nyquist

    b, a = signal.butter(order, [low, high], btype="band")

    filtered = np.zeros_like(data, dtype=np.float64)
    for channel in range(data.shape[0]):
        filtered[channel, :] = signal.filtfilt(b, a, data[channel, :])

    return filtered.astype(data.dtype, copy=False)


def compute_rms(
    data: np.ndarray,
    sampling_rate: float,
    window_ms: float = 100.0,
) -> np.ndarray:
    """
    Compute a moving RMS (root mean square) envelope for each channel.

    Parameters
    ----------
    data : np.ndarray
        Shape (channels, samples).
    sampling_rate : float
        Sampling rate in Hz.
    window_ms : float
        RMS window length in milliseconds. The window is centered
        (symmetric) around each sample and is clipped at the edges of
        the signal.

    Returns
    -------
    np.ndarray
        RMS envelope, same shape as the input.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (channels, samples), got shape {data.shape}")
    if window_ms <= 0:
        raise ValueError("window_ms must be greater than 0.")

    num_channels, num_samples = data.shape

    window_size = max(1, int((window_ms / 1000) * sampling_rate))
    half_window = window_size // 2

    rms = np.zeros_like(data, dtype=np.float64)

    for channel in range(num_channels):
        channel_signal = data[channel, :]

        # Cumulative sum of squares gives an O(n) sliding-window RMS
        # instead of the O(n * window_size) double loop from exercise 2 --
        # this matters here because the live view recomputes RMS
        # repeatedly on a rolling buffer that can hold tens of thousands
        # of samples.
        squared = channel_signal.astype(np.float64) ** 2
        cumsum = np.concatenate(([0.0], np.cumsum(squared)))

        start = np.maximum(0, np.arange(num_samples) - half_window)
        end = np.minimum(num_samples, np.arange(num_samples) + half_window + 1)

        window_sums = cumsum[end] - cumsum[start]
        window_counts = end - start

        rms[channel, :] = np.sqrt(window_sums / window_counts)

    return rms.astype(data.dtype, copy=False)


def apply_signal_mode(
    data: np.ndarray,
    sampling_rate: float,
    mode: str,
    *,
    low_cut: float = 20.0,
    high_cut: float = 450.0,
    filter_order: int = 4,
    rms_window_ms: float = 100.0,
) -> np.ndarray:
    """
    Convenience dispatcher used by the ViewModel to apply the currently
    selected signal mode without needing to import / branch on
    apply_bandpass_filter and compute_rms separately.

    Parameters
    ----------
    data : np.ndarray
        Shape (channels, samples).
    sampling_rate : float
        Sampling rate in Hz.
    mode : str
        One of "original", "rms", "filtered".

    Returns
    -------
    np.ndarray
        Processed data, same shape as input.

    Raises
    ------
    ValueError
        If `mode` is not one of the supported values.
    """
    if mode == "original":
        return data
    elif mode == "filtered":
        return apply_bandpass_filter(
            data, sampling_rate, low_cut=low_cut, high_cut=high_cut, order=filter_order
        )
    elif mode == "rms":
        return compute_rms(data, sampling_rate, window_ms=rms_window_ms)
    else:
        raise ValueError(
            f"Unknown signal mode '{mode}'. Expected one of: original, rms, filtered."
        )