import ast
import threading

import paho.mqtt.client as mqtt
import pandas as pd


class MQTTHandler(object):
    def __init__(self, bridge):
        self.brigde = bridge
        self.client = mqtt.Client("Basestation")
        self._mqtt_thread = threading.Thread(target=self.mqtt_thread)
        self._mqtt_thread.start()

    def on_connect(self, _client, _, __, rc):
        print("Connected to MQTT server")
        self.client.subscribe("drones")
        self.client.subscribe("goals")
        self.client.subscribe("params")
        self.client.subscribe("sensors")

    def handle_mqtt_message(self, _client, _, msg):
        try:
            message = str(msg.payload, 'utf-8')
            if msg.topic == "sensors":
                self.brigde(ast.literal_eval(message), "sensors")
            elif msg.topic == "drones":
                self.brigde(ast.literal_eval(message), "drones")
            elif msg.topic == "goals":
                self.brigde(ast.literal_eval(message), "goals")
            elif msg.topic == "params":
                self.brigde(message, "properties")
        except KeyError as e:
            print("data not understood: ", e)
        except AssertionError as e:
            print("data not valid: ", e)

    def on_message(self, _client, user_data, msg):
        msg_thread = threading.Thread(target=self.handle_mqtt_message, args=(self.client, user_data, msg,))
        msg_thread.start()
        msg_thread.join()

    def send_request(self):
        self.client.publish("step", True)

    def on_disconnect(self, _client, _, rc=0):
        print("Disconnected result code " + str(rc))
        self.client.loop_stop()

    def mqtt_thread(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        # client.username_pw_set("", "")

        # client.connect('192.168.20.22', 1883, 60)
        self.client.connect('127.0.0.1', 1883, 60)
        _running = True
        self.client.loop_forever()


class Database(object):
    sensors_df = None
    drones_df = None
    properties_df = None

    def __init__(self, sensors_dir, drones_dir, properties_dir):
        self.sensors_df = pd.read_csv(sensors_dir)
        self.drones_df = pd.read_csv(drones_dir)
        self.drones_df.set_index('id', inplace=True)
        self.goals_df = pd.read_csv(drones_dir)
        self.goals_df.set_index('id', inplace=True)
        self.properties_df = pd.read_csv(properties_dir)

        self.sensors_c_index = 0
        self.drones_c_index = 0
        self.goals_c_index = 0
        self.properties_c_index = 0

        self.updating_sensors = False
        self.updating_drones = False
        self.updating_goals = False
        self.updating_properties = False
        self.drones_updated = False
        self.goals_updated = False

        self.online_db = MQTTHandler(self.incoming_data)

    def needs_update(self):
        needs_ud = []
        if self.sensors_c_index < self.sensors_df.shape[0] and not self.updating_sensors:
            needs_ud.append('sensors')
        if self.drones_updated and not self.updating_drones:
            needs_ud.append('drones')
            self.drones_updated = False
        if self.goals_updated and not self.updating_goals:
            needs_ud.append('goals')
            self.goals_updated = False
        if self.properties_c_index < self.properties_df.shape[0] and not self.updating_properties:
            needs_ud.append('properties')
        return needs_ud

    def incoming_data(self, new_data, target):
        if target == "sensors":
            self.sensors_df = self.sensors_df.append(new_data, ignore_index=True)
        if target == "drones":
            # print("new", new_data)
            if new_data["id"] in self.drones_df.values:
                auxpd = pd.DataFrame({'id': new_data["id"], 'x': new_data["x"], 'y': new_data["y"]}, index=[0])
                auxpd.set_index('id', inplace=True)
                # print(auxpd)
                self.drones_df.update(auxpd)
            else:
                self.drones_df = self.drones_df.append(new_data, ignore_index=True)
            self.drones_updated = True
        if target == "goals":
            # print("new", new_data)
            if new_data["id"] in self.goals_df.values:
                auxpd = pd.DataFrame({'id': new_data["id"], 'x': new_data["x"], 'y': new_data["y"]}, index=[0])
                auxpd.set_index('id', inplace=True)
                # print(auxpd)
                self.goals_df.update(auxpd)
            else:
                self.goals_df = self.goals_df.append(new_data, ignore_index=True)
            self.goals_updated = True
            # print("db is", self.drones_df)
        if target == "properties":
            self.properties_c_index -= 1
            self.properties_df.at[0, "acq"] = new_data
