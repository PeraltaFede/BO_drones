import os
import sys
import threading
from datetime import datetime
from copy import copy

import matplotlib.cm as cm
import matplotlib.image as img
import numpy as np
import yaml
from Coordinators.informed_coordinator import Coordinator
from GUI.ui_mainwindow import Ui_MainWindow
from PySide2.QtCore import QTimer, Slot, Signal, QObject
from PySide2.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from v2.Database.pandas_database import Database


class SignalManager(QObject):
    sensor_sig = Signal(str)
    drones_sig = Signal(str)
    proper_sig = Signal(str)

    images_ready = Signal(str)


def obtain_map_data(map_yaml_name):
    with open(map_yaml_name, 'r') as stream:
        try:
            map_yaml = yaml.load(stream, yaml.FullLoader)
            map_data = img.imread(os.path.join(os.path.dirname(map_yaml_name), map_yaml.get('image')))
            if map_yaml.get('negate') == 0:
                map_data = np.flipud(map_data[:, :, 0])
                map_data = 1 - map_data
            else:
                map_data = np.flipud(map_data[:, :, 0])
        except yaml.YAMLError:
            map_data = None
        finally:
            return map_data


class GUI(QMainWindow):
    def __init__(self, database: Database):

        super(GUI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.ui.actionGuardar.triggered.connect(self.save_figure)
        self.ui.actionSalir.triggered.connect(self.close)
        self.ui.actionReiniciar.triggered.connect(self.restart_figure)

        self.ui.actionMapa.setChecked(False)
        self.ui.actionFuncion_de_Adquisici_n.setChecked(True)

        self.ui.actionMapa.triggered.connect(self.update_images)
        self.ui.actionPredicci_n_GP.triggered.connect(self.update_images)
        self.ui.actionIncertidumbre_GP.triggered.connect(self.update_images)
        self.ui.actionFuncion_de_Adquisici_n.triggered.connect(self.update_images)
        self.ui.action3D.triggered.connect(self.send_request)

        self.db = database
        initial_map = obtain_map_data(
            "E:/ETSI/Proyecto/data/Map/" + self.db.properties_df['map'].values[0] + "/map.yaml")
        sensors = {}
        for data in self.db.properties_df.values:
            if data[1] not in sensors.keys():
                sensors[data[1]] = initial_map

        wid = QWidget()
        wid.resize(250, 150)
        hbox = QHBoxLayout(wid)

        self.map_fig = Figure(figsize=(7, 5), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_toolbar = NavigationToolbar(self.map_canvas, self)
        self.map_lay = QVBoxLayout()
        self.map_lay.addWidget(self.map_toolbar)
        self.map_lay.addWidget(self.map_canvas)

        self.gp_fig = Figure(figsize=(7, 5), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.gp_canvas = FigureCanvas(self.gp_fig)
        self.gp_toolbar = NavigationToolbar(self.gp_canvas, self)
        self.gp_lay = QVBoxLayout()
        self.gp_lay.addWidget(self.gp_toolbar)
        self.gp_lay.addWidget(self.gp_canvas)

        self.std_fig = Figure(figsize=(7, 5), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.std_canvas = FigureCanvas(self.std_fig)
        self.std_toolbar = NavigationToolbar(self.std_canvas, self)
        self.std_lay = QVBoxLayout()
        self.std_lay.addWidget(self.std_toolbar)
        self.std_lay.addWidget(self.std_canvas)

        self.acq_fig = Figure(figsize=(7, 5), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.acq_canvas = FigureCanvas(self.acq_fig)
        self.acq_toolbar = NavigationToolbar(self.acq_canvas, self)
        self.acq_lay = QVBoxLayout()
        self.acq_lay.addWidget(self.acq_toolbar)
        self.acq_lay.addWidget(self.acq_canvas)

        hbox.addLayout(self.map_lay)
        hbox.addLayout(self.gp_lay)
        hbox.addLayout(self.std_lay)
        hbox.addLayout(self.acq_lay)

        self.map_canvas.setVisible(False)
        self.map_toolbar.setVisible(False)

        # self.acq_canvas.setVisible(False)
        # self.acq_toolbar.setVisible(False)

        self.setCentralWidget(wid)
        wid.show()

        self.image = []
        self.data = []
        self.titles = []
        self.nans = []
        self.shape = None
        i = 1
        for key in sensors:
            ax = self.map_fig.add_subplot(111)
            # plt.subplot(row, col, i)
            self.image.append(ax.imshow(sensors[key], origin='lower'))
            self.titles.append("{} real".format(key))
            ax.set_title("Map for sensor {}".format(key))
            self.data.append(sensors[key])

            for k in range(np.shape(self.data[0])[0]):
                for j in range(np.shape(self.data[0])[1]):
                    if self.data[0][k, j] == 1.0:
                        self.nans.append([k, j])

            ax = self.gp_fig.add_subplot(111)
            self.image.append(ax.imshow(sensors[key], origin='lower'))
            self.titles.append("{} gp".format(key))
            ax.set_title("Sensor {} Gaussian Process Regression".format(key))
            self.data.append(sensors[key])
            self.gp_fig.colorbar(self.image[i], ax=ax, orientation='horizontal')
            current_cmap = copy(cm.get_cmap())
            current_cmap.set_bad(color='white')

            ax = self.std_fig.add_subplot(111)
            self.image.append(ax.imshow(sensors[key], origin='lower'))
            self.titles.append("{} gp un".format(key))
            ax.set_title("Sensor {} GP Uncertainty".format(key))
            self.data.append(sensors[key])
            self.gp_fig.colorbar(self.image[i + 1], ax=ax, orientation='horizontal')
            current_cmap = copy(cm.get_cmap())
            current_cmap.set_bad(color='white')

            ax = self.acq_fig.add_subplot(111)
            self.image.append(ax.imshow(sensors[key], origin='lower'))
            self.titles.append("{} acq".format(key))
            ax.set_title("Sensor {} Acquisition Function".format(key))
            self.data.append(sensors[key])
            self.gp_fig.colorbar(self.image[i + 2], ax=ax, orientation='horizontal')
            current_cmap = copy(cm.get_cmap())
            current_cmap.set_bad(color='white')

            i += 4
            if self.shape is None:
                self.shape = np.shape(sensors[key])

        self.row = None
        self.col = None
        self.vmin = 0
        self.vmax = 0
        self.sm = None

        self.coordinator = Coordinator(None, self.data[0], 't')

        self.update_thread = QTimer()
        self.update_thread.setInterval(250)
        self.update_thread.timeout.connect(self.update_request)
        self.update_thread.start()

        self.signalManager = SignalManager()
        self.signalManager.sensor_sig.connect(self.request_sensors_update)
        self.signalManager.drones_sig.connect(self.request_drones_update)
        # self.signalManager.proper_sig.connect(self.request_proper_update)
        self.signalManager.images_ready.connect(self.update_images)

    @Slot()
    def update_request(self):
        for updatable_data in self.db.needs_update():
            if "sensors" in updatable_data:
                self.signalManager.sensor_sig.emit('')
            elif "drones" in updatable_data:
                self.signalManager.drones_sig.emit('')
            elif "properties" in updatable_data:
                self.signalManager.proper_sig.emit('')

    @Slot()
    def request_sensors_update(self):
        # print('updating sensors')
        threading.Thread(target=self.generate_sensor_image, ).start()

    @Slot()
    def request_drones_update(self):
        # print('updating drones')
        threading.Thread(target=self.update_drone_position).start()

    @Slot()
    def request_proper_update(self):
        print('updating proper')

    def update_drone_position(self):
        self.db.updating_drones = True
        self.db.drones_c_index += 1
        self.db.updating_drones = False

    def generate_sensor_image(self):
        self.db.updating_sensors = True

        raw_data = self.db.sensors_df.loc[
            self.db.sensors_df['type'] == 't'].to_numpy()
        last_index = len(raw_data)
        raw_data = raw_data[self.db.sensors_c_index:, 1:4]
        new_data = [[row[:2], row[2]] for row in raw_data]
        if self.db.sensors_c_index == 0:
            self.coordinator.initialize_data_gpr(new_data)
        else:
            self.coordinator.add_data(new_data[0])
        self.coordinator.fit_data()
        self.db.sensors_c_index = last_index

        observe_maps = dict()

        if self.ui.actionPredicci_n_GP.isChecked() or self.ui.actionIncertidumbre_GP.isChecked():
            if self.ui.actionIncertidumbre_GP.isChecked():
                mu, std, sensor_name = self.coordinator.surrogate(return_std=True,
                                                                  return_sensor=True)
            else:
                mu, sensor_name = self.coordinator.surrogate(return_std=False,
                                                             return_sensor=True)
            if self.ui.actionPredicci_n_GP.isChecked():
                observe_maps["{} gp".format(sensor_name)] = mu
            if self.ui.actionIncertidumbre_GP.isChecked():
                observe_maps["{} gp un".format(sensor_name)] = std

        if self.ui.actionFuncion_de_Adquisici_n.isChecked():
            acq, sensor_name = self.coordinator.get_acq()
            observe_maps["{} acq".format(sensor_name)] = acq

        self.observe_maps(observe_maps)
        self.db.updating_sensors = False

    def observe_maps(self, images: dict):
        for i in range(len(self.titles)):
            if self.titles[i] in images.keys():
                for key in images.keys():
                    if self.titles[i] == key:
                        data = images[key].reshape((self.shape[1], self.shape[0])).T
                        for nnan in self.nans:
                            data[nnan[0], nnan[1]] = -1
                        # todo: plt.contour(self.environment.render_maps()["t"], colors='k', alpha=0.3, linewidths=1.0)

                        self.data[i] = np.ma.array(data, mask=(data == -1))

                        self.image[i].set_data(self.data[i])
                        self.image[i].set_clim(vmin=np.min(self.data[i]), vmax=np.max(self.data[i]))
                        # a.update_ticks()
                        break

        self.signalManager.images_ready.emit('')

    def restart_figure(self):
        self.coordinator.data = [np.array([[], []]), np.array([])]
        self.db.sensors_df = self.db.sensors_df.iloc[0:0]
        self.db.sensors_c_index = 0

        for i in range(len(self.titles)):
            self.image[i].set_data(self.data[0])
            self.image[i].set_clim(vmin=np.min(self.data[0]), vmax=np.max(self.data[0]))
        self.signalManager.images_ready.emit('')
        # threading.Thread(target=self.observe_maps, args=(,),).start()

        # self.coordinator = Coordinator(None, self.data[0], 't')

    @Slot()
    def update_images(self):

        if self.ui.actionMapa.isChecked():
            self.map_canvas.draw()
            if not self.map_canvas.isVisible():
                self.map_canvas.setVisible(True)
                self.map_toolbar.setVisible(True)
        else:
            if self.map_canvas.isVisible():
                self.map_canvas.setVisible(False)
                self.map_toolbar.setVisible(False)

        if self.ui.actionPredicci_n_GP.isChecked():
            self.gp_canvas.draw()
            if not self.gp_canvas.isVisible():
                self.gp_canvas.setVisible(True)
                self.gp_toolbar.setVisible(True)
        else:
            if self.gp_canvas.isVisible():
                self.gp_canvas.setVisible(False)
                self.gp_toolbar.setVisible(False)

        if self.ui.actionIncertidumbre_GP.isChecked():
            self.std_canvas.draw()
            if not self.std_canvas.isVisible():
                self.std_canvas.setVisible(True)
                self.std_toolbar.setVisible(True)
        else:
            if self.std_canvas.isVisible():
                self.std_canvas.setVisible(False)
                self.std_toolbar.setVisible(False)

        if self.ui.actionFuncion_de_Adquisici_n.isChecked():
            self.acq_canvas.draw()
            if not self.acq_canvas.isVisible():
                self.acq_canvas.setVisible(True)
                self.acq_toolbar.setVisible(True)
        else:
            if self.acq_canvas.isVisible():
                self.acq_canvas.setVisible(False)
                self.acq_toolbar.setVisible(False)

    def export_maps(self, extension='png'):
        for my_image, my_title in zip(self.data, self.titles):
            if self.row is None:
                self.row, self.col = np.where(np.isnan(my_image))
                self.sm = cm.ScalarMappable(cmap='viridis')
                self.vmin = np.min(my_image)
                self.vmax = np.max(my_image)
            if "un" in my_title:
                self.sm.set_clim(np.min(my_image), np.max(my_image))
            my_image[self.row, self.col] = 0
            new_image = self.sm.to_rgba(my_image, bytes=True)
            new_image[self.row, self.col, :] = [0, 0, 0, 0]
            new_image = np.flipud(new_image)
            img.imsave("E:/ETSI/Proyecto/results/Map/{}_{}.{}".format(datetime.now().timestamp(), my_title, extension),
                       new_image)
            # plt.show(block=True)

    def send_request(self):
        self.db.online_db.send_request()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    db = Database(sensors_dir="E:/ETSI/Proyecto/data/Databases/CSV/Sensors.csv",
                  drones_dir="E:/ETSI/Proyecto/data/Databases/CSV/Drones.csv",
                  properties_dir="E:/ETSI/Proyecto/data/Databases/CSV/Properties.csv")

    window = GUI(db)

    window.show()

    sys.exit(app.exec_())
