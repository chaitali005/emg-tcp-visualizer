from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from views.plotView import VisPyPlotWidget


class MainView(QMainWindow):
    """
    Main application window (View).

    Responsibilities:
    - Own all visible widgets:
        * TCP port input + Connect / Disconnect buttons
        * channel selector
        * signal mode selector (original / filtered / RMS)
        * Y scale control
        * "Plot All Channels" button
        * "Open Offline Plot" button
        * Start / Stop live plotting button
        * status label
        * signal time label
        * VisPy plot widget
    - Connect user interactions to the ViewModel's public API.
    - React to ViewModel signals to update visible state.

    The View does NOT handle TCP or raw signal processing directly.
    """

    def __init__(self, view_model):
        super().__init__()

        self.view_model = view_model

        self.setWindowTitle("TCP EMG Viewer")
        self.resize(1200, 800)

        # ------------------------------------------------------------------
        # Central widget / main layout
        # ------------------------------------------------------------------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ------------------------------------------------------------------
        # Top: signal time label
        # ------------------------------------------------------------------
        self.time_label = QLabel("Signal time: 0.00 s")
        self.time_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(self.time_label)

        # ------------------------------------------------------------------
        # Middle: controls (left) + plot (right)
        # ------------------------------------------------------------------
        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)
        main_layout.addLayout(content_layout)

        control_layout = QVBoxLayout()
        control_layout.setSpacing(8)

        # ---------------- TCP connection controls ----------------
        self.port_label = QLabel("TCP Port")
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("e.g. 12345")

        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")

        port_row = QHBoxLayout()
        port_row.addWidget(self.port_label)
        port_row.addWidget(self.port_input)

        conn_buttons_row = QHBoxLayout()
        conn_buttons_row.addWidget(self.connect_button)
        conn_buttons_row.addWidget(self.disconnect_button)

        # ---------------- Channel selection ----------------
        self.channel_label = QLabel("Channel")
        self.channel_combo = QComboBox()
        # 32 channels; data is 0-based, label is 1-based
        for ch in range(32):
            self.channel_combo.addItem(f"Channel {ch + 1}", ch)

        # ---------------- Signal mode selection ----------------
        self.mode_label = QLabel("Signal mode")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Original", "original")
        self.mode_combo.addItem("Filtered", "filtered")
        self.mode_combo.addItem("RMS", "rms")

        # ---------------- Y scale control ----------------
        self.y_scale_label = QLabel("Y scale")
        self.y_scale_input = QDoubleSpinBox()
        self.y_scale_input.setRange(0.01, 100000.0)
        self.y_scale_input.setValue(300.0)
        self.y_scale_input.setSingleStep(50.0)
        self.y_scale_input.setDecimals(2)

        # ---------------- Plot buttons ----------------
        self.toggle_button = QPushButton("Start Plotting")
        self.plot_all_button = QPushButton("Plot All Channels")
        self.offline_button = QPushButton("Open Offline Plot")

        # ---------------- Status / info label ----------------
        self.info_label = QLabel("Start the TCP server first.")

        # Add controls to the left control layout
        control_layout.addLayout(port_row)
        control_layout.addLayout(conn_buttons_row)
        control_layout.addWidget(self.channel_label)
        control_layout.addWidget(self.channel_combo)
        control_layout.addWidget(self.mode_label)
        control_layout.addWidget(self.mode_combo)
        control_layout.addWidget(self.y_scale_label)
        control_layout.addWidget(self.y_scale_input)
        control_layout.addWidget(self.plot_all_button)
        control_layout.addWidget(self.offline_button)
        control_layout.addStretch()
        control_layout.addWidget(self.info_label)
        control_layout.addWidget(self.toggle_button)

        # ---------------- Plot widget on the right ----------------
        self.plot_widget = VisPyPlotWidget(
            visible_duration_seconds=10.0,
            y_scale=self.y_scale_input.value(),
        )

        content_layout.addLayout(control_layout, stretch=0)
        content_layout.addWidget(self.plot_widget, stretch=1)

        # ------------------------------------------------------------------
        # Connections between View and ViewModel
        # ------------------------------------------------------------------

        # Start / Stop live plotting
        self.toggle_button.clicked.connect(self.toggle_plotting)

        # Y scale → VisPy widget
        self.y_scale_input.valueChanged.connect(self.plot_widget.set_y_scale)

        # TCP connect / disconnect
        self.connect_button.clicked.connect(self.on_connect_clicked)
        self.disconnect_button.clicked.connect(self.view_model.disconnect)

        # Channel and mode selection
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        # Plot all channels
        self.plot_all_button.clicked.connect(self.view_model.plot_all_channels)

        # Open offline Matplotlib view
        self.offline_button.clicked.connect(self.view_model.open_offline_plot)

        # ViewModel → View signals
        self.view_model.plot_updated.connect(self.plot_widget.update_plot)
        self.view_model.status_updated.connect(self.info_label.setText)
        self.view_model.signal_time_updated.connect(self.update_signal_time)
        self.view_model.signal_time_updated.connect(self.plot_widget.set_signal_time)

    # ------------------------------------------------------------------
    # Slots / handlers for user actions
    # ------------------------------------------------------------------

    def toggle_plotting(self):
        """
        Start or stop live plotting using the ViewModel.
        """
        if self.view_model.is_plotting:
            self.view_model.stop_plotting()
            self.toggle_button.setText("Start Plotting")
        else:
            self.view_model.start_plotting()
            if self.view_model.is_plotting:
                self.toggle_button.setText("Stop Plotting")

    def on_connect_clicked(self):
        """
        Handler for the 'Connect' button. Reads the port from the line edit
        and calls view_model.connect(port).
        """
        text = self.port_input.text().strip()
        if not text:
            self.info_label.setText("Please enter a TCP port.")
            return

        try:
            port = int(text)
        except ValueError:
            self.info_label.setText("Invalid port. Please enter a number.")
            return

        self.view_model.connect(port)

    def on_channel_changed(self, index: int):
        """
        Handler for the channel combo box.
        """
        channel = self.channel_combo.itemData(index)
        if channel is not None:
            self.view_model.set_channel(channel)

    def on_mode_changed(self, index: int):
        """
        Handler for the signal mode combo box.
        """
        mode = self.mode_combo.itemData(index)
        if mode is not None:
            self.view_model.set_mode(mode)

    def update_signal_time(self, signal_time_seconds: float):
        """
        Update the time label whenever the ViewModel emits a new signal time.
        """
        self.time_label.setText(f"Signal time: {signal_time_seconds:.2f} s")
