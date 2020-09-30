# TODO: mejorar el gui, mas titulos, selectores

import os
import sys
import threading
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.image as img
import numpy as np
import yaml
from PySide2.QtCore import QTimer, Slot, Signal, QObject
from PySide2.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from bin.Coordinators.informed_coordinator import Coordinator
from bin.GUI.ui_mainwindow import Ui_MainWindow
from bin.v2.Database.pandas_database import Database


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
        self.db = database
        initial_map = obtain_map_data(
            "E:/ETSI/Proyecto/data/Map/" + self.db.properties_df['map'].values[0] + "/map.yaml")
        sensors = {}
        for data in self.db.properties_df.values:
            if data[1] not in sensors.keys():
                sensors[data[1]] = initial_map

        self.fig = Figure(figsize=(640, 480), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.canvas = FigureCanvas(self.fig)
        self.setCentralWidget(self.canvas)

        row = np.ceil(np.sqrt(len(sensors)))
        col = np.round(np.sqrt(len(sensors))) * 3

        self.image = []
        self.data = []
        self.titles = []
        self.nans = []
        self.shape = None
        i = 1
        for key in sensors:
            ax = self.fig.add_subplot(int(row * 100 + col * 10 + i))
            # plt.subplot(row, col, i)
            self.image.append(ax.imshow(sensors[key], origin='lower'))
            self.titles.append("{} real".format(key))
            ax.set_title("Map for sensor {}".format(key))
            self.data.append(sensors[key])

            for k in range(np.shape(self.data[0])[0]):
                for j in range(np.shape(self.data[0])[1]):
                    if self.data[0][k, j] == 1.0:
                        self.nans.append([k, j])

            ax = self.fig.add_subplot(int(row * 100 + col * 10 + i + 1))
            # plt.subplot(row, col, i + 1)
            self.image.append(ax.imshow(sensors[key], origin='lower'))
            self.titles.append("{} gp".format(key))
            ax.set_title("Sensor {} Gaussian Process Regression".format(key))
            self.data.append(sensors[key])
            # todo:shrink cb
            self.fig.colorbar(self.image[i], ax=ax, orientation='horizontal')
            current_cmap = cm.get_cmap()
            current_cmap.set_bad(color='white')

            ax = self.fig.add_subplot(int(row * 100 + col * 10 + i + 2))
            # plt.subplot(row, col, i + 2)
            self.image.append(ax.imshow(sensors[key], origin='lower'))
            self.titles.append("{} gp un".format(key))
            ax.set_title("Sensor {} GP Uncertainty".format(key))
            self.data.append(sensors[key])
            self.fig.colorbar(self.image[i + 1], ax=ax, orientation='horizontal')
            current_cmap = cm.get_cmap()
            current_cmap.set_bad(color='white')

            i += 3
            if self.shape is None:
                self.shape = np.shape(sensors[key])

        self.row = None
        self.col = None
        self.vmin = 0
        self.vmax = 0
        self.sm = None

        # todo connect to a server instead of creating a new object every time
        self.coordinator = Coordinator(None, self.data[0], 't')

        self.update_thread = QTimer()
        self.update_thread.setInterval(500)
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
        print('updating drones')
        threading.Thread(target=self.update_drone_position).start()
        # TODO:investigar si puedo usar mpl como GUI, ventajas con respecto a pyside2

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
        mu, std, sensor_name = self.coordinator.surrogate(return_std=True,
                                                          return_sensor=True)
        self.observe_maps({"{} gp".format(sensor_name): mu, "{} gp un".format(sensor_name): std})
        self.db.updating_sensors = False

    def observe_maps(self, images: dict):
        for key in images.keys():
            for i in range(len(self.titles)):
                if self.titles[i] == key:
                    data = images[key].reshape(self.shape[1], self.shape[0]).T
                    for nnan in self.nans:
                        data[nnan[0], nnan[1]] = -1

                    self.data[i] = np.ma.array(data, mask=(data == -1))

                    self.image[i].set_data(self.data[i])
                    self.image[i].set_clim(vmin=np.min(self.data[i]), vmax=np.max(self.data[i]))
                    # a.update_ticks()
                    break

        self.signalManager.images_ready.emit('')

    @Slot()
    def update_images(self):
        self.canvas.draw()

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    db = Database(sensors_dir="E:/ETSI/Proyecto/data/Databases/CSV/Sensors.csv",
                  drones_dir="E:/ETSI/Proyecto/data/Databases/CSV/Drones.csv",
                  properties_dir="E:/ETSI/Proyecto/data/Databases/CSV/Properties.csv")

    window = GUI(db)

    window.show()

    sys.exit(app.exec_())
