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

    # Vertical offset between channels:
    # use overall max abs as scale, but ensure it's not too small
    max_abs = float(np.max(np.abs(all_y))) if all_y.size > 0 else 1.0
    if max_abs <= 0:
        max_abs = 1.0
    offset = max_abs * 2.0  # slightly larger separation

    plt.figure(figsize=(10, 6))

    for ch in range(n_channels):
        # Optional: remove per-channel mean so they don't drift too far
        y = all_y[ch, :]
        y_centered = y - np.mean(y)
        y_shifted = y_centered + ch * offset
        plt.plot(x, y_shifted, label=f"Ch {ch + 1}")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + offset")
    plt.title("All Channels (stacked)")
    # plt.legend(loc="upper right", ncol=2, fontsize=8)  # optional
    plt.tight_layout()
    plt.show()

