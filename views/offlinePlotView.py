import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
)

from models.signal_processing import apply_signal_mode


class OfflinePlotView(QDialog):
    """
    Simple offline inspection dialog using Matplotlib.

    Shows one channel at a time; allows selecting:
    - channel
    - signal mode (original / filtered / RMS)
    """

    def __init__(self, data, sampling_rate, parent=None):
        super().__init__(parent)

        # data shape: (channels, samples)
        self.data = data
        self.sampling_rate = sampling_rate
        self.mode = "original"
        self.channel = 0

        self.setWindowTitle("Offline EMG Inspection")

        layout = QVBoxLayout(self)

        # Controls row
        controls = QHBoxLayout()
        layout.addLayout(controls)

        # Channel selector
        controls.addWidget(QLabel("Channel"))
        self.channel_combo = QComboBox()
        for ch in range(self.data.shape[0]):
            self.channel_combo.addItem(f"Channel {ch + 1}", ch)
        controls.addWidget(self.channel_combo)

        # Mode selector
        controls.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Original", "original")
        self.mode_combo.addItem("Filtered", "filtered")
        self.mode_combo.addItem("RMS", "rms")
        controls.addWidget(self.mode_combo)

        # Buttons
        self.plot_button = QPushButton("Plot")
        controls.addWidget(self.plot_button)

        self.close_button = QPushButton("Close")
        controls.addWidget(self.close_button)

        # Connections
        self.plot_button.clicked.connect(self.plot)
        self.close_button.clicked.connect(self.close)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

    def on_channel_changed(self, index):
        ch = self.channel_combo.itemData(index)
        if ch is not None:
            self.channel = ch

    def on_mode_changed(self, index):
        mode = self.mode_combo.itemData(index)
        if mode is not None:
            self.mode = mode

    def update_data(self, data):
        """Update the recording data when new data has arrived."""
        self.data = data

    def plot(self):
        """Plot the selected channel and mode using Matplotlib."""
        if self.data is None or self.data.size == 0:
            return

        y = self.data[self.channel, :].reshape(1, -1)
        y_proc = apply_signal_mode(
            y,
            sampling_rate=self.sampling_rate,
            mode=self.mode,
        )[0, :]

        t = np.arange(y_proc.size) / self.sampling_rate

        plt.figure(figsize=(10, 4))
        plt.plot(t, y_proc)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Channel {self.channel + 1} - {self.mode}")
        plt.tight_layout()
        plt.show()
