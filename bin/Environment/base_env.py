import os

import matplotlib.image as img
import numpy as np
import yaml


class BaseEnv(object):
    def __init__(self, map_path2yaml):
        self.grid = np.array([[], []])
        self.resolution = 0
        self.origin = np.zeros((3, 1))
        self.obtain_map_data(map_path2yaml)

    def obtain_map_data(self, map_yaml_name):
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
                self.grid = map_data
                self.resolution = map_yaml.get('resolution')
                self.origin = map_yaml.get('origin')
