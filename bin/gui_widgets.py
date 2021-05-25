import os
import sys
import threading
from copy import copy
from time import time

import matplotlib.cm as cm
import matplotlib.image as img
import numpy as np
import yaml
from Coordinators.multi_informed_coordinator import Coordinator
from GUI.ui_mainwindow import Ui_MainWindow
from PySide2.QtCore import QTimer, Slot, Signal, QObject
from PySide2.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QVBoxLayout
from Utils.voronoi_regions import calc_voronoi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from bin.v2.Database.pandas_database import Database


class SignalManager(QObject):
    sensor_sig = Signal(str)
    drones_sig = Signal(str)
    goals_sig = Signal(str)
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
        self.ui.actionAutomatic.triggered.connect(self.change_automatic)

        self.auto_BO = True

        self.db = database
        initial_map = obtain_map_data(
            "E:/ETSI/Proyecto/data/Map/" + self.db.properties_df['map'].values[0] + "/map.yaml")
        sensors = {}
        for data in self.db.properties_df.values:
            if data[1] not in sensors.keys():
                sensors[data[1]] = initial_map

        self.acq_func = self.db.properties_df["acq"][0]

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
        self.nans = np.load(open('/data/Databases/CSV/nans.npy', 'rb'))
        self.shape = None
        self.axes = []
        self.drone_pos_ax = []
        self.drone_goals_ax = dict()
        self.voronoi_ax = dict()
        self.signal_from_gp_std = False

        xticks = np.arange(0, 1000, 200)
        yticks = np.arange(0, 1500, 200)
        xnticks = [str(num * 10) for num in xticks]
        ynticks = [str(num * 10) for num in yticks]
        self.colors = ["#3B4D77", "#C09235", "#B72F56", "#91B333", "#FFD100"]

        i = 1
        for key in sensors:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            cmap = plt.cm.coolwarm
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(1, 0, cmap.N)
            my_cmap = ListedColormap(my_cmap)
            ax = self.map_fig.add_subplot(111)
            # plt.subplot(row, col, i)
            self.image.append(ax.imshow(sensors[key], origin='lower', cmap=my_cmap, zorder=5))
            ax.grid(True, zorder=0, color="white")
            ax.set_facecolor('#eaeaf2')
            ax.tick_params(axis="both", labelsize=30)
            ax.set_ylabel('y [m]', fontsize=30)
            ax.set_xlabel('x [m]', fontsize=30)
            ax.set_xticks(xticks)  # values
            ax.set_xticklabels(xnticks)  # labels
            ax.set_yticks(yticks)  # values
            ax.set_yticklabels(ynticks)  # labels
            self.titles.append("{} real".format(key))
            # ax.set_title("Map for sensor {}".format(key))
            self.data.append(sensors[key])
            self.axes.append(ax)

            # return base.from_list(cmap_name, color_list, N)

            ax = self.gp_fig.add_subplot(111)
            # cm.get_cmap(base_cmap, N)
            current_cmap = copy(cm.get_cmap("plasma"))
            current_cmap.set_bad(color="#00000000")
            self.image.append(
                ax.imshow(sensors[key], origin='lower', cmap=current_cmap, zorder=5))
            ax.grid(True, zorder=0, color="white")
            ax.set_facecolor('#eaeaf2')
            self.titles.append("{} gp".format(key))
            # ax.set_title("Sensor {} Gaussian Process Regression".format(key))
            ax.tick_params(axis="both", labelsize=30)
            # ax.set_ylabel('y [m]', fontsize=30)
            ax.set_xlabel('x [m]', fontsize=30)
            ax.set_xticks(xticks)  # values
            ax.set_xticklabels(xnticks)  # labels
            ax.set_yticks(yticks)  # values
            ax.set_yticklabels(ynticks)  # labels
            self.data.append(sensors[key])
            self.image[-1].set_clim(vmin=-2.586, vmax=1.7898)
            cb = self.gp_fig.colorbar(self.image[i], ax=ax)
            cb.ax.tick_params(labelsize=20)
            cb.ax.set_xlabel(r'$SE(\mu (x))$', fontsize=30)
            self.axes.append(ax)

            ax = self.std_fig.add_subplot(111)
            current_cmap = copy(cm.get_cmap("summer"))
            current_cmap.set_bad(color="#00000000")
            self.image.append(ax.imshow(sensors[key], origin='lower', cmap=current_cmap, zorder=5, vmin=0.0, vmax=1.0))
            ax.grid(True, zorder=0, color="white")
            ax.set_facecolor('#eaeaf2')
            self.titles.append("{} gp un".format(key))
            # ax.set_title("Sensor {} GP Uncertainty".format(key))
            ax.tick_params(axis="both", labelsize=30)
            # ax.set_ylabel('y [m]', fontsize=30)
            ax.set_xlabel('x [m]', fontsize=30)
            ax.set_xticks(xticks)  # values
            ax.set_xticklabels(xnticks)  # labels
            ax.set_yticks(yticks)  # values
            ax.set_yticklabels(ynticks)  # labels
            self.data.append(sensors[key])
            # self.image[-1].set_clim(vmin=0.0, vmax=1.0)
            cb = self.gp_fig.colorbar(self.image[i + 1], ax=ax)
            cb.ax.tick_params(labelsize=20)
            cb.ax.set_xlabel(r'$\sigma (x)$', fontsize=30)
            self.axes.append(ax)

            ax = self.acq_fig.add_subplot(111)
            current_cmap = copy(cm.get_cmap("YlGn_r"))
            current_cmap.set_bad(color="#00000000")
            self.image.append(ax.imshow(sensors[key], origin='lower', cmap=current_cmap, zorder=5))
            ax.grid(True, zorder=0, color="white")
            ax.set_facecolor('#eaeaf2')
            self.titles.append("{} acq".format(key))
            # ax.set_title("Sensor {} Acquisition Function".format(key))
            ax.tick_params(axis="both", labelsize=30)
            # ax.set_ylabel('y', fontsize=30)
            ax.set_xlabel('x [m]', fontsize=30)
            ax.set_xticks(xticks)  # values
            ax.set_xticklabels(xnticks)  # labels
            ax.set_yticks(yticks)  # values
            ax.set_yticklabels(ynticks)  # labels

            self.data.append(sensors[key])
            # cb = self.gp_fig.colorbar(self.image[i + 2], ax=ax, orientation='horizontal')
            # cb.ax.set_xlabel(r'$\mathrm{\mathsf{SEI}} (x)$')
            self.axes.append(ax)

            i += 4
            if self.shape is None:
                self.shape = np.shape(sensors[key])

        self.row = None
        self.col = None
        self.vmin = 0
        self.vmax = 0
        self.sm = None

        self.coordinator = Coordinator(self.data[0], 't')
        self.real_map = np.load("E:/ETSI/Proyecto/data/Databases/numpy_files/random_12.npy")

        self.update_thread = QTimer()
        self.update_thread.setInterval(250)
        self.update_thread.timeout.connect(self.update_request)
        self.update_thread.start()

        self.signalManager = SignalManager()
        self.signalManager.sensor_sig.connect(self.request_sensors_update)
        self.signalManager.drones_sig.connect(self.request_drones_update)
        self.signalManager.goals_sig.connect(self.request_goals_update)
        self.signalManager.proper_sig.connect(self.request_proper_update)
        self.signalManager.images_ready.connect(self.update_images)
        self.current_drone_pos = np.zeros((1, 2))

    @Slot()
    def change_automatic(self):
        self.auto_BO = self.ui.actionAutomatic.isChecked()
        self.ui.action3D.setEnabled(not self.auto_BO)
        if self.auto_BO and self.db.sensors_c_index > 0:
            self.send_request()

    @Slot()
    def update_request(self):
        for updatable_data in self.db.needs_update():
            if "sensors" in updatable_data:
                self.signalManager.sensor_sig.emit('')
            elif "drones" in updatable_data:
                self.signalManager.drones_sig.emit('')
            elif "goals" in updatable_data:
                self.signalManager.goals_sig.emit('')
            elif "properties" in updatable_data:
                self.signalManager.proper_sig.emit('')

    @Slot()
    def request_sensors_update(self):
        threading.Thread(target=self.generate_sensor_image, ).start()

    @Slot()
    def request_drones_update(self):
        threading.Thread(target=self.update_drone_position).start()

    @Slot()
    def request_goals_update(self):
        # print("new goal received, pls uncomment")
        threading.Thread(target=self.update_goals_position).start()

    @Slot()
    def request_proper_update(self):
        self.acq_func = self.db.properties_df["acq"][0]
        self.db.properties_c_index += 1
        print('updating acq to, ', self.acq_func)

    def update_goals_position(self):
        self.db.updating_goals = True
        poses = self.db.goals_df.to_numpy()
        for read in poses:
            current_drone_goals = np.array(read[:2])

            # clr = 'g' if int(read[2]) == 0 else 'k' if int(read[2]) == 1 else 'm' if int(read[2]) == 2 else 'y'
            if read[2] not in self.drone_goals_ax.keys():
                self.drone_goals_ax[read[2]] = []
                if self.ui.actionPredicci_n_GP.isChecked():
                    self.drone_goals_ax[read[2]].append(
                        self.axes[1].plot(current_drone_goals[0], current_drone_goals[1], 'X',
                                          color=self.colors[int(read[2])],
                                          markersize=12, zorder=6))  # , label="ID: {}".format(int(read[2]))
                    # self.axes[1].legend(loc="lower left", prop={"size": 20})
                if self.ui.actionIncertidumbre_GP.isChecked():
                    self.drone_goals_ax[read[2]].append(
                        self.axes[2].plot(current_drone_goals[0], current_drone_goals[1], 'X',
                                          color=self.colors[int(read[2])],
                                          markersize=12, zorder=6))
                # if self.ui.actionFuncion_de_Adquisici_n.isChecked():
                #     self.drone_goals_ax[read[2]].append(
                #         self.axes[3].plot(current_drone_goals[0], current_drone_goals[1], 'X',
                #                           color=self.colors[int(read[2])],
                #                           markersize=12, zorder=6))
            else:
                if self.ui.actionPredicci_n_GP.isChecked():
                    self.drone_goals_ax[read[2]][0][0].set_xdata(current_drone_goals[0])
                    self.drone_goals_ax[read[2]][0][0].set_ydata(current_drone_goals[1])
                if self.ui.actionIncertidumbre_GP.isChecked():
                    self.drone_goals_ax[read[2]][1][0].set_xdata(current_drone_goals[0])
                    self.drone_goals_ax[read[2]][1][0].set_ydata(current_drone_goals[1])
                # if self.ui.actionFuncion_de_Adquisici_n.isChecked():
                #     self.drone_goals_ax[read[2]][2][0].set_xdata(current_drone_goals[0])
                #     self.drone_goals_ax[read[2]][2][0].set_ydata(current_drone_goals[1])
        self.db.goals_c_index += 1
        self.signalManager.images_ready.emit('')
        self.db.updating_goals = False

    def update_drone_position(self):
        self.db.updating_drones = True
        poses = self.db.drones_df.to_numpy()
        self.db.drones_c_index += 1
        # print(poses)
        # print(len(self.drone_pos_ax), len(self.db.drones_df.index))
        for read in poses:
            current_drone_pos = np.array(read[:2])
            if len(self.drone_pos_ax) < len(self.db.drones_df.index) * 2:
                if self.ui.actionPredicci_n_GP.isChecked():
                    self.drone_pos_ax.append(
                        self.axes[1].plot(current_drone_pos[0], current_drone_pos[1], '.',
                                          color=self.colors[int(read[2])], markersize=12,
                                          zorder=7, label="ID: {}".format(int(read[2]))))
                    self.axes[1].legend(loc="lower left", prop={"size": 20}).set_zorder(10)
                if self.ui.actionIncertidumbre_GP.isChecked():
                    self.drone_pos_ax.append(
                        self.axes[2].plot(current_drone_pos[0], current_drone_pos[1], '.',
                                          color=self.colors[int(read[2])], markersize=12,
                                          zorder=7))
                    # self.axes[1].legend(loc="lower left", prop={"size": 20})
                # if self.ui.actionFuncion_de_Adquisici_n.isChecked():
                #     self.drone_pos_ax.append(
                #         self.axes[3].plot(current_drone_pos[0], current_drone_pos[1], '.',
                #                           color=self.colors[int(read[2])], markersize=12,
                #                           zorder=7))
            else:
                if self.ui.actionPredicci_n_GP.isChecked():
                    self.drone_pos_ax[int(read[2] * 2)][0].set_xdata(current_drone_pos[0])
                    self.drone_pos_ax[int(read[2] * 2)][0].set_ydata(current_drone_pos[1])
                if self.ui.actionIncertidumbre_GP.isChecked():
                    self.drone_pos_ax[int(1 + read[2] * 2)][0].set_xdata(current_drone_pos[0])
                    self.drone_pos_ax[int(1 + read[2] * 2)][0].set_ydata(current_drone_pos[1])
                    # self.axes[1].legend(loc="lower left", prop={"size": 20})
                # if self.ui.actionFuncion_de_Adquisici_n.isChecked():
                #     self.drone_pos_ax[int(2 + read[2] * 3)][0].set_xdata(current_drone_pos[0])
                #     self.drone_pos_ax[int(2 + read[2] * 3)][0].set_ydata(current_drone_pos[1])
                # self.axes[2].plot(current_drone_pos[0], current_drone_pos[1], '^{}'.format(clr), markersize=12,
                #                   zorder=6)
        self.signalManager.images_ready.emit('')
        self.db.updating_drones = False

    def generate_sensor_image(self):
        self.db.updating_sensors = True

        raw_data = self.db.sensors_df.loc[
            self.db.sensors_df['type'] == 's1'].to_numpy()
        last_index = len(raw_data)
        raw_data = raw_data[self.db.sensors_c_index:, 1:6]

        # if self.db.sensors_c_index == 0:
        #     new_data = [[row[:2], row[2]] for row in raw_data]
        #     self.coordinator.initialize_data_gpr(new_data)

        #     if self.ui.actionPredicci_n_GP.isChecked():
        #         self.axes[0].plot(new_data[0][0][0], new_data[0][0][1], 'Dy', markersize=12,
        #                           label="Initial Information", zorder=6)
        #         self.axes[0].plot(new_data[1][0][0], new_data[1][0][1], 'Dy', markersize=12, zorder=6)
        #         self.axes[0].plot(new_data[2][0][0], new_data[2][0][1], '^y', markersize=12, zorder=6,
        #                           label="Previous Positions")
        #     if self.ui.actionIncertidumbre_GP.isChecked():
        #         self.axes[1].plot(new_data[0][0][0], new_data[0][0][1], 'Dy', markersize=12,
        #                           label="Initial Information", zorder=6)
        #         self.axes[1].plot(new_data[1][0][0], new_data[1][0][1], 'Dy', markersize=12, zorder=6)
        #         self.axes[1].plot(new_data[2][0][0], new_data[2][0][1], '^y', markersize=12,
        #                           label="Previous Positions", zorder=6)
        #     if self.ui.actionFuncion_de_Adquisici_n.isChecked():
        #         self.axes[2].plot(new_data[0][0][0], new_data[0][0][1], 'Dy', markersize=12, zorder=6)
        #         self.axes[2].plot(new_data[1][0][0], new_data[1][0][1], 'Dy', markersize=12, zorder=6)
        #         self.axes[2].plot(new_data[2][0][0], new_data[2][0][1], '^y', markersize=12, zorder=6)
        # else:
        for data in raw_data:
            if self.db.sensors_c_index > 0:
                poses = self.db.drones_df.to_numpy()
                _, reg = calc_voronoi(data[:2], [np.array(read[:2]) for read in poses if read[2] != data[4]],
                                      self.data[0])
            else:
                _, reg = calc_voronoi(data[:2], [np.array(read[:2]) for read in raw_data if read[4] != data[4]],
                                      self.data[0])
            reg = np.vstack([reg, reg[0, :]])

            if self.colors[int(data[4])] in self.voronoi_ax.keys():
                for line in self.voronoi_ax[self.colors[int(data[4])]]:
                    line.pop(0).remove()
            self.voronoi_ax[self.colors[int(data[4])]] = []

            if self.ui.actionMapa.isChecked():
                # self.axes[0].plot(data[0], data[1], '^{}'.format(drone_color), markersize=12,
                #                   zorder=6, alpha=0.7, label="Drone: {}".format(int(data[4])))
                self.axes[0].plot(data[0], data[1], '^', color=self.colors[int(data[4])], markersize=12,
                                  zorder=6, alpha=0.7, label="Drone: {}".format(int(data[4])))
                self.voronoi_ax[self.colors[int(data[4])]].append(
                    self.axes[0].plot(reg[:, 0], reg[:, 1], '-', color=self.colors[int(data[4])], zorder=9, lw=3))
                # self.axes[0].legend(loc="lower left", prop={"size": 20})
            if self.ui.actionPredicci_n_GP.isChecked():
                self.axes[1].plot(data[0], data[1], '^', color=self.colors[int(data[4])], markersize=12,
                                  zorder=6, alpha=0.7)
                self.voronoi_ax[self.colors[int(data[4])]].append(
                    self.axes[1].plot(reg[:, 0], reg[:, 1], '-', color=self.colors[int(data[4])], zorder=9))
            if self.ui.actionIncertidumbre_GP.isChecked():
                self.axes[2].plot(data[0], data[1], '^', color=self.colors[int(data[4])], markersize=12,
                                  zorder=6, alpha=0.7)
                self.voronoi_ax[self.colors[int(data[4])]].append(
                    self.axes[2].plot(reg[:, 0], reg[:, 1], '-', color=self.colors[int(data[4])], zorder=9))
            # if self.ui.actionFuncion_de_Adquisici_n.isChecked():
            #     self.axes[3].plot(data[0], data[1], '^', color=self.colors[int(data[4])], markersize=12,
            #                       zorder=6, alpha=0.7)
            #     self.voronoi_ax[self.colors[int(data[4])]].append(
            #         self.axes[3].plot(reg[:, 0], reg[:, 1], '-', color=self.colors[int(data[4])], zorder=9))
            self.coordinator.add_data([data[:2], data[2]])
        self.coordinator.fit_data()
        self.db.sensors_c_index = last_index

        self.current_drone_pos = raw_data[-1][0:3]

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
                # observe_maps["{} gp".format(sensor_name)] = (mu - self.real_map) ** 2
            if self.ui.actionIncertidumbre_GP.isChecked():
                observe_maps["{} gp un".format(sensor_name)] = std

        if self.ui.actionFuncion_de_Adquisici_n.isChecked():
            acq, sensor_name = self.coordinator.get_acq(self.current_drone_pos, self.acq_func)
            observe_maps["{} acq".format(sensor_name)] = acq

        self.observe_maps(observe_maps)
        self.db.updating_sensors = False

        if self.auto_BO:
            self.send_request()

    def observe_maps(self, images: dict):
        for i in range(len(self.titles)):
            if self.titles[i] in images.keys():
                for key in images.keys():
                    if self.titles[i] == key:
                        data = images[key].reshape((self.shape[1], self.shape[0])).T
                        # if "t gp" == key:
                        #     data = np.power(data - self.real_map, 2)
                        for nnan in self.nans:
                            data[nnan[0], nnan[1]] = -1

                        self.data[i] = np.ma.array(data, mask=(data == -1))

                        self.image[i].set_data(self.data[i])
                        if key == "t acq":
                            self.image[i].set_clim(vmin=np.min(self.data[i]), vmax=np.max(self.data[i]))
                        # if "acq" in key:
                        #     ipos, kpos = np.where(self.data[i] == np.nanmax(self.data[i]))
                        #     print(ipos, kpos)
                        #     self.axes.plot(kpos[0], ipos[0], 'Xr', markersize=10)
                        # a.update_ticks()
                        break

        self.signalManager.images_ready.emit('')
        self.signal_from_gp_std = True

    def restart_figure(self):
        self.coordinator.data = [np.array([[], []]), np.array([])]
        self.db.sensors_df = self.db.sensors_df.iloc[0:0]
        self.db.sensors_c_index = 0

        for i in range(len(self.titles)):
            self.image[i].set_data(self.data[0])
            self.image[i].set_clim(vmin=np.min(self.data[0]), vmax=np.max(self.data[0]))
        for ax in self.axes:
            while len(ax.lines) > 0:
                [line.remove() for line in ax.lines]

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
        if self.signal_from_gp_std:
            self.gp_fig.savefig("E:/ETSI/Papers/congreso 2021/multridrones/err_{}.png".format(time()))
            self.std_fig.savefig("E:/ETSI/Papers/congreso 2021/multridrones/std_{}.png".format(time()))
            self.signal_from_gp_std = False

    #
    # def export_maps(self, extension='png'):
    #     for my_image, my_title in zip(self.data, self.titles):
    #         if self.row is None:
    #             self.row, self.col = np.where(np.isnan(my_image))
    #             self.sm = cm.ScalarMappable(cmap='viridis')
    #             self.vmin = np.min(my_image)
    #             self.vmax = np.max(my_image)
    #         if "un" in my_title:
    #             self.sm.set_clim(np.min(my_image), np.max(my_image))
    #         my_image[self.row, self.col] = 0
    #         new_image = self.sm.to_rgba(my_image, bytes=True)
    #         new_image[self.row, self.col, :] = [0, 0, 0, 0]
    #         new_image = np.flipud(new_image)
    #         img.imsave("E:/ETSI/Proyecto/results/Map/{}_{}.{}".format(datetime.now().timestamp(), my_title, extension),
    #                    new_image)
    #         # plt.show(block=True)

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
