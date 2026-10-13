from PyQt5 import QtWidgets
from window import Ui_MainWindow
from training_window import TrainingWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_learn.clicked.connect(self.open_training)
        self.training_window = None

    def open_training(self):
        if self.training_window is None:
            self.training_window = TrainingWindow()
        self.training_window.show()
        self.training_window.raise_()