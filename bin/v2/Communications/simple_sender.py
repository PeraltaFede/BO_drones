import time

import paho.mqtt.client as mqtt


class Sender(object):
    def __init__(self):
        self.client = mqtt.Client("sender")
        self.client.connect('127.0.0.1', 1883, 60)
        self.ids = ["type", "longitude", "latitude", "value", "altitude", "drone", "timestamp"]

    def send_new_sensor_msg(self, raw_x_y_val):
        msg = "t," + raw_x_y_val + ",1, 2, 3"
        payload = dict(zip(self.ids, msg.split(",")))
        for key in payload:
            try:
                payload[key] = float(payload[key])
            except Exception as e:
                pass
        self.client.publish("sensors", str(payload))
        # print("Message {} sent".format(payload))

    def send_new_drone_msg(self, raw_x_y):
        msg = "{},{},{},{},{},{}".format("d_0", raw_x_y[0], raw_x_y[1], 0, time.time(), "online")
        payload = dict(zip(self.ids, msg.split(",")))
        for key in payload:
            try:
                payload[key] = float(payload[key])
            except Exception as e:
                pass
        self.client.publish("drones", str(payload))
