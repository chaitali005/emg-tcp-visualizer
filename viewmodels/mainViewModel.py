from PySide6.QtCore import QObject, QTimer, Signal

from models.tcp_client_model import TcpClientModel
from models.signal_processing import apply_signal_mode
from models.recording_buffer import RecordingBuffer
from views.offlinePlotView import OfflinePlotView
from views.allChannelsView import plot_all_channels_matplotlib


class MainViewModel(QObject):
    """
    ViewModel for the TCP EMG live plotting application.

    Responsibilities:
    - manage the TcpClientModel (connect / disconnect / receive data)
    - maintain a RecordingBuffer for offline inspection
    - run a QTimer that periodically fetches new data
    - apply the selected signal mode (original / filtered / RMS)
    - emit x/y plot data to the View
    - emit current signal time and status messages to the View
    """

    # Signals to the View
    plot_updated = Signal(object, object)      # x, y
    status_updated = Signal(str)              # status text
    signal_time_updated = Signal(float)       # seconds

    def __init__(self):
        super().__init__()

        # Default connection parameters
        self.host = "localhost"
        self.port = 12345

        # Should match the server / exercise spec
        self.sampling_rate = 2000.0  # Hz
        self.channels = 32
        self.samples_per_packet = 18
        self.window_seconds = 10.0

        # Current signal mode: "original", "filtered", or "rms"
        self.mode = "original"

        # Create the TCP client model
        self.model = TcpClientModel(
            host=self.host,
            port=self.port,
            sampling_rate=self.sampling_rate,
            channels=self.channels,
            samples_per_packet=self.samples_per_packet,
            window_seconds=self.window_seconds,
            selected_channel=0,  # 0-based index; Channel 1 in the UI
        )

        # Buffer for offline recording
        self.recording_buffer = RecordingBuffer(
            channels=self.channels,
            sampling_rate=self.sampling_rate,
        )
        # Reference to the offline view dialog (created lazily)
        self.offline_view: OfflinePlotView | None = None

        self.is_plotting = False

        # Timer to poll for new data and update the plot
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)

    # ------------------------------------------------------------------
    # Connection API (used by the View)
    # ------------------------------------------------------------------

    def connect(self, port: int) -> None:
        """
        Connect to the TCP server on the given port.
        Called by MainView when the user clicks 'Connect'.
        """
        # Store port and update model's port before connecting
        self.port = port
        self.model.port = port

        try:
            self.model.connect()
        except OSError as error:
            self.status_updated.emit(f"Could not connect to server: {error}")
            return

        # Clear any previous recording when starting a new session
        self.recording_buffer.clear()

        self.status_updated.emit(f"Connected to TCP server on port {port}.")

    def disconnect(self) -> None:
        """
        Disconnect from the server and stop plotting, if active.
        Called by MainView when the user clicks 'Disconnect'.
        """
        if self.is_plotting:
            self.timer.stop()
            self.is_plotting = False

        self.model.disconnect()
        self.status_updated.emit("Disconnected from TCP server.")

    # ------------------------------------------------------------------
    # Plot control (used by Start/Stop button in the View)
    # ------------------------------------------------------------------

    def start_plotting(self) -> None:
        """
        Start updating the plot.
        If not yet connected, try to connect using the current port.
        """
        if self.is_plotting:
            return

        # Ensure we are connected
        if not self.model.is_connected:
            try:
                self.model.connect()
            except OSError as error:
                self.status_updated.emit(f"Could not connect to server: {error}")
                return

        self.is_plotting = True
        self.status_updated.emit("Live plotting started.")
        # Timer interval: 10 ms (adjust as needed)
        self.timer.start(10)

    def stop_plotting(self) -> None:
        """
        Stop updating the plot and close the TCP connection.
        """
        if not self.is_plotting and not self.model.is_connected:
            return

        self.timer.stop()
        self.model.disconnect()
        self.is_plotting = False
        self.status_updated.emit("Disconnected from TCP server.")

    # ------------------------------------------------------------------
    # Data / plot update
    # ------------------------------------------------------------------

    def update_plot(self) -> None:
        """
        Receive new TCP data and emit updated plot data.

        Called repeatedly by the QTimer while live plotting is active.
        """
        # Pull in any new bytes and update the rolling buffer
        new_data = self.model.receive_data()

        # Append new data to the offline recording buffer
        if new_data is not None:
            # new_data shape: (channels, n_new_samples)
            self.recording_buffer.append(new_data)

        if not self.model.has_data():
            return

        # Get current rolling window for the selected channel
        x, y = self.model.get_window()  # y is 1D (n_samples,)

        # Apply the selected signal mode
        # Convert to (channels, samples) for processing, then back
        y_channel = y.reshape(1, -1)
        y_processed = apply_signal_mode(
            y_channel,
            sampling_rate=self.sampling_rate,
            mode=self.mode,
        )
        y_out = y_processed[0, :]

        # Emit data to the View
        self.plot_updated.emit(x, y_out)

        # Emit current signal time based on received samples
        signal_time = self.model.get_signal_time_seconds()
        self.signal_time_updated.emit(signal_time)

    # ------------------------------------------------------------------
    # Channel and mode selection (used by combo boxes in the View)
    # ------------------------------------------------------------------

    def set_channel(self, channel: int) -> None:
        """
        Update which channel is shown in the live plot.
        Called by MainView when the channel combo changes.

        Parameters
        ----------
        channel : int
            0-based index of the channel (0..31).
        """
        try:
            self.model.set_selected_channel(channel)
            self.status_updated.emit(f"Selected channel: {channel + 1}")
        except ValueError as error:
            self.status_updated.emit(str(error))

    def set_mode(self, mode: str) -> None:
        """
        Update the current signal mode ("original", "filtered", "rms").
        Called by MainView when the mode combo changes.
        """
        if mode not in ("original", "filtered", "rms"):
            self.status_updated.emit(
                f"Unknown mode '{mode}'. Expected original, filtered, or rms."
            )
            return

        self.mode = mode
        self.status_updated.emit(f"Signal mode set to: {mode}")

    # ------------------------------------------------------------------
    # Plot-all-channels hook
    # ------------------------------------------------------------------

    def plot_all_channels(self) -> None:
        """
        Called by the 'Plot All Channels' button in the View.

        Shows all channels in a stacked Matplotlib plot for the current
        rolling window from TcpClientModel, using the current signal mode.
        """
        if not self.model.has_data():
            self.status_updated.emit(
                "No data available yet to plot all channels."
            )
            return

        x, all_y = self.model.get_all_channels_window()

        # Apply current mode to all channels (original / filtered / RMS)
        all_y_processed = apply_signal_mode(
            all_y,
            sampling_rate=self.sampling_rate,
            mode=self.mode,
        )

        plot_all_channels_matplotlib(x, all_y_processed, self.sampling_rate)
        self.status_updated.emit("Plotted all channels in a Matplotlib window.")


    # ------------------------------------------------------------------
    # Offline Matplotlib inspection
    # ------------------------------------------------------------------

    def open_offline_plot(self) -> None:
        """
        Open a Matplotlib window for offline inspection of the recorded signal.
        Uses the RecordingBuffer contents.
        """
        if not self.recording_buffer.has_data():
            self.status_updated.emit("No data available for offline plotting.")
            return

        _, full_data = self.recording_buffer.get_all_channels()

        # Lazy-create the offline view once and reuse it
        if self.offline_view is None:
            self.offline_view = OfflinePlotView(
                data=full_data,
                sampling_rate=self.sampling_rate,
            )
        else:
            self.offline_view.update_data(full_data)

        self.offline_view.show()
        self.status_updated.emit("Opened offline Matplotlib view.")
