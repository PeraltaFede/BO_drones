import sys
import threading

from PySide2.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from bin.GUI.ui_mainwindow import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


class GUI(object):
    def __init__(self):
        self.thread = threading.Thread(target=self.app_exec)
        self.thread.start()

    def app_exec(self):
        self.app = QApplication(sys.argv)

        self.fig = Figure(figsize=(640, 480), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = self.fig.add_subplot(111)
        # generate the canvas to display the plot
        self.canvas = FigureCanvas(self.fig)

        self.window = MainWindow()
        self.window.setCentralWidget(self.canvas)

        self.window.show()
        self.app.exec_()

    def update_plot(self, image):
        self.axes.imshow(image)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    fig = Figure(figsize=(640, 480), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
    ax = fig.add_subplot(111)
    ax.plot([0, 1])
    # generate the canvas to display the plot
    canvas = FigureCanvas(fig)

    window = MainWindow()
    window.setCentralWidget(canvas)

    window.show()

    sys.exit(app.exec_())
