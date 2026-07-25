"""
RecordingBuffer: stores the full history of received samples for offline
inspection (Matplotlib view), separate from TcpClientModel's rolling
10-second live buffer.

The two buffers serve different purposes:

- TcpClientModel.data_buffer: short rolling window, only used for the
  live VisPy plot. Old samples are discarded to keep memory and plotting
  cost bounded.
- RecordingBuffer.data: grows for the entire duration of the connection
  (or up to an optional cap) so that, once the user disconnects, the
  complete recorded signal can still be inspected channel-by-channel in
  the offline Matplotlib view.

This module has no GUI or networking dependencies; it only knows how to
store and retrieve NumPy arrays.
"""

import numpy as np


class RecordingBuffer:
    """
    Accumulates received signal chunks for offline inspection.

    Parameters
    ----------
    channels : int
        Number of channels (rows) the buffer should expect.
    sampling_rate : float
        Sampling rate in Hz, used to compute a time axis for plotting.
    dtype : np.dtype
        Storage dtype. Should match the dtype produced by
        TcpClientModel (float64).
    max_samples : int or None
        Optional safety cap on the number of stored samples per channel.
        If set and the buffer would exceed this size, the oldest samples
        are dropped (oldest-first), similar to the live rolling buffer
        but with a much larger window intended to cover a realistic
        recording length. If None, the buffer grows without bound for
        the lifetime of the connection.
    """

    def __init__(
        self,
        channels: int,
        sampling_rate: float,
        dtype=np.float64,
        max_samples: int | None = None,
    ):
        if channels <= 0:
            raise ValueError("channels must be a positive integer.")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive number.")

        self.channels = channels
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.max_samples = max_samples

        self.data = np.empty((channels, 0), dtype=dtype)

    def append(self, new_data: np.ndarray) -> None:
        """
        Append newly received samples to the recording.

        Parameters
        ----------
        new_data : np.ndarray
            Shape (channels, n_new_samples). Typically this is exactly
            the array returned by TcpClientModel.receive_data().

        Raises
        ------
        ValueError
            If new_data does not have the expected number of channels,
            or is not 2-dimensional.
        """
        if new_data is None:
            return

        if new_data.ndim != 2:
            raise ValueError(
                f"new_data must be 2D (channels, samples), got shape {new_data.shape}"
            )
        if new_data.shape[0] != self.channels:
            raise ValueError(
                f"new_data has {new_data.shape[0]} channels, "
                f"expected {self.channels}."
            )
        if new_data.shape[1] == 0:
            return

        self.data = np.concatenate((self.data, new_data.astype(self.dtype, copy=False)), axis=1)

        if self.max_samples is not None and self.data.shape[1] > self.max_samples:
            self.data = self.data[:, -self.max_samples:]

    def has_data(self) -> bool:
        """Return True if at least one sample has been recorded."""
        return self.data.shape[1] > 0

    def num_samples(self) -> int:
        """Return the number of samples currently stored per channel."""
        return self.data.shape[1]

    def duration_seconds(self) -> float:
        """Return the duration of the recorded signal in seconds."""
        return self.num_samples() / self.sampling_rate

    def get_channel(self, channel: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the time axis and signal for a single channel across the
        whole recording.

        Parameters
        ----------
        channel : int
            Channel index, 0-based.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t, y) where t is the time axis in seconds and y is the
            recorded signal for that channel.

        Raises
        ------
        ValueError
            If channel is out of range, or no data has been recorded yet.
        """
        if not (0 <= channel < self.channels):
            raise ValueError(f"channel must be in range 0..{self.channels - 1}, got {channel}")
        if not self.has_data():
            raise ValueError("No data has been recorded yet.")

        y = self.data[channel, :]
        t = np.arange(y.shape[0]) / self.sampling_rate
        return t, y

    def get_all_channels(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the time axis and the full (channels, samples) array for
        the whole recording, e.g. for an offline "all channels" view.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t, data) where t is the time axis in seconds and data has
            shape (channels, samples).

        Raises
        ------
        ValueError
            If no data has been recorded yet.
        """
        if not self.has_data():
            raise ValueError("No data has been recorded yet.")

        t = np.arange(self.data.shape[1]) / self.sampling_rate
        return t, self.data

    def clear(self) -> None:
        """Discard all recorded data, e.g. when starting a new connection."""
        self.data = np.empty((self.channels, 0), dtype=self.dtype)