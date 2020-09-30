# import threading
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np


class GUI(object):
    def __init__(self, maps: dict, show_readings=True):

        row = np.ceil(np.sqrt(len(maps)))
        col = np.round(np.sqrt(len(maps))) * 3

        self.image = []
        self.data = []
        self.titles = []
        self.shape = None
        self.show_readings = show_readings
        i = 1
        for key in maps:
            plt.subplot(row, col, i)
            self.image.append(plt.imshow(maps[key], origin='lower'))
            self.titles.append("{} real".format(key))
            self.data.append(maps[key])
            plt.title("{} real".format(key))
            plt.colorbar()
            plt.subplot(row, col, i + 1)
            self.image.append(plt.imshow(maps[key], origin='lower'))
            self.titles.append("{} gp".format(key))
            self.data.append(maps[key])
            plt.title("{} gp".format(key))
            plt.colorbar()
            plt.subplot(row, col, i + 2)
            self.image.append(plt.imshow(maps[key], origin='lower'))
            self.titles.append("{} gp un".format(key))
            self.data.append(maps[key])
            plt.title("{} gp un".format(key))
            plt.colorbar()
            i += 3
            if self.shape is None:
                self.shape = np.shape(maps[key])

        self.row = None
        self.col = None
        self.vmin = 0
        self.vmax = 0
        self.sm = None

        plt.pause(0.01)
        plt.draw()
        # plt.show(block=True)

    # self.thread = threading.Thread(target=self.app_exec)
    #     self.thread.start()

    def observe_maps(self, data: dict):
        for key in data.keys():
            for i in range(len(self.titles)):
                if self.titles[i] == key:
                    self.data[i] = data[key].reshape(self.shape[1], self.shape[0]).T
                    self.image[i].set_data(self.data[i])
                    break
        plt.draw()
        plt.pause(0.001)

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
            plt.show(block=True)
