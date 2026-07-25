from PySide6.QtWidgets import QApplication, QLabel
import sys

app = QApplication(sys.argv)
label = QLabel("Qt test window")
label.resize(200, 50)
label.show()
sys.exit(app.exec())
