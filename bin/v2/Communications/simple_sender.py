import time
import threading

import paho.mqtt.client as mqtt


class Sender(object):
    def __init__(self):
        self.client = mqtt.Client("sender")

        self._mqtt_thread = threading.Thread(target=self.mqtt_thread)
        self._mqtt_thread.start()

        self.step = True

        self.ids = ["type", "longitude", "latitude", "value", "altitude", "drone", "timestamp"]

    def mqtt_thread(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect('127.0.0.1', 1883, 60)
        self.client.loop_forever()

    def on_connect(self, _client, _, __, rc):
        print("Connected to MQTT server")
        self.client.subscribe("step")

    def should_update(self):
        if self.step:
            self.step = False
            return True
        else:
            return False

    def on_message(self, _client, user_data, msg):
        message = bool(msg.payload)
        if msg.topic == "step":
            self.step = message

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
