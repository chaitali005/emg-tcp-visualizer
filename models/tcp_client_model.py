"""
TcpClientModel: connects to the provided EMG TCP server and reconstructs
the streamed signal into a rolling NumPy buffer for live plotting.

Data contract (must match the server exactly):

    channels            = 32
    samples_per_packet  = 18
    dtype               = float64
    packet size         = 32 * 18 * 8 = 4608 bytes

The server sends one packet per "window" of the recording, roughly every
samples_per_packet / sampling_rate seconds. TCP is a byte stream, so a
single recv() call is not guaranteed to contain exactly one packet --
bytes are accumulated in self.byte_buffer and only converted into
complete (channels, samples_per_packet) packets once enough bytes have
arrived.

This module is intentionally free of any GUI/Qt imports. It is used by
the ViewModel, but does not know about it.
"""

import socket
import numpy as np


class TcpClientModel:
    """
    TCP client for receiving streamed EMG data and maintaining a rolling
    buffer for live plotting.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port.
    sampling_rate : float
        Sampling rate in Hz, used to compute time axes and signal time.
    channels : int
        Number of channels per packet.
    samples_per_packet : int
        Number of samples per channel in a single packet.
    window_seconds : float
        Length of the rolling live buffer, in seconds. Older samples are
        discarded as new ones arrive. This only affects the live view --
        it has no effect on offline inspection, which is handled by a
        separate RecordingBuffer fed from receive_data()'s return value.
    selected_channel : int
        Initially selected channel for get_window().
    """

    def __init__(
        self,
        host: str,
        port: int,
        sampling_rate: float,
        channels: int,
        samples_per_packet: int,
        window_seconds: float,
        selected_channel: int = 0,
    ):
        if not (0 <= selected_channel < channels):
            raise ValueError(
                f"selected_channel must be in range 0..{channels - 1}, "
                f"got {selected_channel}"
            )

        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.samples_per_packet = samples_per_packet
        self.window_seconds = window_seconds
        self.selected_channel = selected_channel

        # Must match the dtype the server uses before calling .tobytes().
        self.dtype = np.float64

        self.socket: socket.socket | None = None
        self.is_connected = False

        self.packet_size = self.channels * self.samples_per_packet
        self.packet_size_bytes = self.packet_size * np.dtype(self.dtype).itemsize

        self.window_size = int(self.sampling_rate * self.window_seconds)

        self.byte_buffer = bytearray()
        self.data_buffer = np.empty((self.channels, 0), dtype=self.dtype)

        # Total samples received since connecting. Used for signal_time
        # and is independent of the rolling window cutoff above.
        self.total_samples_received = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Connect to the TCP server.

        Raises
        ------
        OSError
            If the connection cannot be established (e.g. the server is
            not running, the port is wrong, or the host is unreachable).
            ConnectionRefusedError, TimeoutError, and socket.gaierror are
            all subclasses of OSError, so callers can catch OSError alone
            to handle all of these uniformly, as suggested in the project
            README:

                try:
                    self.model.connect()
                except OSError as error:
                    self.status_updated.emit(f"Could not connect: {error}")
        """
        if self.is_connected:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.host, self.port))
        except OSError:
            # Make sure we don't leak a half-open socket if connect()
            # fails partway through (e.g. DNS resolves but the port
            # refuses the connection).
            sock.close()
            raise

        # Non-blocking mode means recv() never freezes the Qt event loop
        # while waiting for data.
        sock.setblocking(False)

        self.socket = sock
        self.is_connected = True

    def disconnect(self) -> None:
        """
        Close the TCP connection, if any.

        Safe to call multiple times and safe to call even if never
        connected -- it will simply do nothing in that case.
        """
        self.is_connected = False

        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                # Already closed / broken socket -- nothing more to do.
                pass
            finally:
                self.socket = None

    # ------------------------------------------------------------------
    # Receiving data
    # ------------------------------------------------------------------

    def receive_data(self) -> np.ndarray | None:
        """
        Receive all currently available TCP data, reconstruct complete
        packets, and append them to the rolling live buffer.

        This method is designed to be called repeatedly (e.g. from a
        QTimer tick in the ViewModel). It never raises on a broken
        connection -- instead it disconnects internally and returns None,
        so the caller does not need to wrap every call in a try/except.

        Returns
        -------
        np.ndarray or None
            The newly decoded data for this call, shape
            (channels, n_new_samples), or None if no complete packet was
            available yet (e.g. not enough bytes have arrived, or the
            connection just closed).

            This return value is intended to be forwarded by the caller
            into a separate RecordingBuffer for offline inspection, e.g.:

                new_data = self.model.receive_data()
                if new_data is not None:
                    self.recording_buffer.append(new_data)
        """
        if not self.is_connected or self.socket is None:
            return None

        while True:
            try:
                new_bytes = self.socket.recv(4096)

                if not new_bytes:
                    # An empty read means the server closed the
                    # connection gracefully.
                    self.disconnect()
                    return None

                self.byte_buffer.extend(new_bytes)

            except BlockingIOError:
                # No more data available right now -- this is the normal
                # case on most ticks, not an error.
                break
            except OSError:
                # Connection lost unexpectedly (e.g. ConnectionResetError,
                # server process killed, network interruption). Treat
                # this the same as a graceful close so the rest of the
                # application can recover instead of crashing.
                self.disconnect()
                return None

        return self._extract_packets_from_buffer()

    def _extract_packets_from_buffer(self) -> np.ndarray | None:
        """
        Convert complete byte packets currently sitting in self.byte_buffer
        into NumPy arrays, append them to the rolling buffer, and return
        the newly decoded data.

        Returns
        -------
        np.ndarray or None
            Shape (channels, n_new_samples) if at least one complete
            packet was extracted, otherwise None.
        """
        packets = []

        while len(self.byte_buffer) >= self.packet_size_bytes:
            packet_bytes = self.byte_buffer[: self.packet_size_bytes]
            del self.byte_buffer[: self.packet_size_bytes]

            packet = np.frombuffer(packet_bytes, dtype=self.dtype)
            packet = packet.reshape(self.channels, self.samples_per_packet)

            packets.append(packet)

        if len(packets) == 0:
            return None

        new_data = np.concatenate(packets, axis=1)

        self.data_buffer = np.concatenate((self.data_buffer, new_data), axis=1)
        self.total_samples_received += new_data.shape[1]

        # Keep only the newest `window_size` samples for the live view.
        if self.data_buffer.shape[1] > self.window_size:
            self.data_buffer = self.data_buffer[:, -self.window_size :]

        return new_data

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------

    def set_selected_channel(self, channel: int) -> None:
        """
        Change which channel get_window() returns.

        Parameters
        ----------
        channel : int
            Channel index, 0-based.

        Raises
        ------
        ValueError
            If channel is out of range.
        """
        if not (0 <= channel < self.channels):
            raise ValueError(
                f"channel must be in range 0..{self.channels - 1}, got {channel}"
            )
        self.selected_channel = channel

    # ------------------------------------------------------------------
    # Reading data out for plotting
    # ------------------------------------------------------------------

    def has_data(self) -> bool:
        """Return True if enough data is available for plotting."""
        return self.data_buffer.shape[1] >= 2

    def get_window(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return x and y data for the live single-channel plot.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            x: relative time axis for the visible rolling window.
            y: the currently selected channel's data, shape (n_samples,).
        """
        y = self.data_buffer[self.selected_channel, :]
        x = np.arange(y.shape[0]) / self.sampling_rate
        return x, y

    def get_all_channels_window(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return x and y data for the live "Plot All Channels" view.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            x: relative time axis for the visible rolling window.
            y: all channels, shape (channels, n_samples). The View is
            responsible for any vertical offset used to make the
            stacked channels readable -- this method only provides the
            raw rolling buffer so the View never has to reach into
            internal state like data_buffer directly.
        """
        n_samples = self.data_buffer.shape[1]
        x = np.arange(n_samples) / self.sampling_rate
        return x, self.data_buffer

    def get_signal_time_seconds(self) -> float:
        """
        Return the total signal time received so far, in seconds.

        This is derived from the total sample count rather than a
        wall-clock timer, so it reflects actual received data even if
        the GUI briefly lags behind real time.
        """
        return self.total_samples_received / self.sampling_rate