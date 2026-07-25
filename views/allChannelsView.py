import numpy as np
import matplotlib.pyplot as plt


def plot_all_channels_matplotlib(x, all_y, sampling_rate: float):
    """
    Plot all channels with vertical offsets using Matplotlib.

    Parameters
    ----------
    x : np.ndarray
        Time axis for the current window, shape (n_samples,).
    all_y : np.ndarray
        All channels data, shape (channels, n_samples).
    sampling_rate : float
        Sampling rate in Hz (for labeling only).
    """
    if all_y is None or all_y.size == 0:
        return

    n_channels, n_samples = all_y.shape

    # Vertical offset between channels
    max_abs = float(np.max(np.abs(all_y))) if all_y.size > 0 else 1.0
    offset = max_abs * 1.2 if max_abs > 0 else 1.0

    plt.figure(figsize=(10, 6))

    for ch in range(n_channels):
        y_shifted = all_y[ch, :] + ch * offset
        plt.plot(x, y_shifted, label=f"Ch {ch + 1}")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + offset")
    plt.title("All Channels (stacked)")
    # Optionally show legend; can be commented out if cluttered
    # plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()
