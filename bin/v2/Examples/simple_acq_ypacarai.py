from sys import path

path.extend([path[0][:path[0].rindex("bin") - 1]])
from bin.Coordinators.gym_coordinator import Coordinator
from bin.Environment.base_env import BaseEnv


class YpacaraiBO(object):
    screen_x0 = -57.38106014642857
    screen_xscale = 1.030035714285734E-4
    screen_y0 = -25.38342902978723
    screen_yscale = 9.33042553191497E-5

    def __init__(self, sensors, path2yaml=path[-1] + "/data/Map/Ypacarai/map.yaml"):
        # ponemos la lista de sensores que se tendrán en cuenta para determinar posiciones
        self.sensors = sensors
        # creamos el mapa
        self.env = BaseEnv(path2yaml)
        # creamos el coordinador encargado de obtener las posiciones de medicion
        self.coordinador = Coordinator(self.env.grid, set(self.sensors))

    def _gps2pix(self, latlng):
        if latlng is None:
            return np.zeros((1, 3))
        screen_y = (latlng["lat"] - self.screen_y0) / self.screen_yscale
        screen_x = (latlng["lon"] - self.screen_x0) / self.screen_xscale
        return [np.round(screen_x).astype(np.int64), np.round(screen_y).astype(np.int64)]

    def _pix2gps(self, screen_pos):
        screen_x = screen_pos[0]
        screen_y = screen_pos[1]
        world_y = self.screen_y0 + self.screen_yscale * screen_y
        world_x = self.screen_x0 + self.screen_xscale * screen_x
        return {"lat": world_y, "lon": world_x}

    def obtener_siguiente_posicion(self, data, pos, other_pos=None):

        new_data = []
        for d in data:
            aux_d = dict()
            for key in d.keys():
                if key != "pos":
                    aux_d[key] = d[key]
            aux_d["pos"] = self._gps2pix(d["pos"])
            new_data.append(aux_d)
        self.coordinador.initialize_data_gpr(new_data)
        return self._pix2gps(
            self.coordinador.generate_new_goal(self._gps2pix(pos), self._gps2pix(other_pos)))


if __name__ == "__main__":
    import numpy as np

    # Sensores
    sensores = ["pH"]
    # Clase de YpacaraiBO
    ypa_bo = YpacaraiBO(sensores)

    # ejemplo de posicion
    pos = {"lat": -25.316577, "lon": -57.316350}
    # ejemplo de un elemento de la lista de datos obtenidos
    una_lectura = dict()
    for sensor in sensores:
        una_lectura[sensor] = np.random.rand() - 0.5
    una_lectura["pos"] = {"lat": pos["lat"] + 0.001 * np.random.rand() - 0.0005,
                          "lon": pos["lon"] + 0.001 * np.random.rand() - 0.0005}

    datos = []
    for i in range(10):
        datos.append(una_lectura)

        # Funcion de obtencion de sgte posicion!
        final_pos = ypa_bo.obtener_siguiente_posicion(datos, pos)

        # generación de informacion de ejemplo
        for sensor in sensores:
            una_lectura[sensor] = np.random.rand() - 0.5
        una_lectura["pos"] = final_pos
