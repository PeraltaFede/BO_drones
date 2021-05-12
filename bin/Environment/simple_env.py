from copy import deepcopy

import bin.Utils.utils as utils
from bin.Environment.base_env import BaseEnv


class Env(BaseEnv):
    def __init__(self, map_path2yaml):
        super().__init__(map_path2yaml)
        self.maps = {}

    def add_new_map(self, sensors, file=-1, clone4noiseless=False):
        for sensor in sensors:
            self.maps[sensor] = utils.create_map(self.grid, self.resolution, True, sensor=sensor, file=file,
                                                 noiseless=True)
            if clone4noiseless:
                self.maps[f"noiseless_{sensor}"] = utils.create_map(self.grid, self.resolution, True, sensor=sensor,
                                                                    file=file, noiseless=True)

    def render_maps(self, sensors=None):
        data = {}
        if sensors is None:
            sensors = self.maps.keys()
        for sensor in sensors:
            if sensor in self.maps.keys():
                data[sensor] = deepcopy(self.maps[sensor])
            else:
                data[sensor] = None
        data["map"] = self.grid
        return data
