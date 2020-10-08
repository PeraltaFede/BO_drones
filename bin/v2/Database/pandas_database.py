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
        self.client.subscribe("sensors")

    def handle_mqtt_message(self, _client, _, msg):
        try:
            message = str(msg.payload, 'utf-8')
            if msg.topic == "sensors":
                self.brigde(ast.literal_eval(message), "sensors")
            elif msg.topic == "drones":
                self.brigde(ast.literal_eval(message), "drones")
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
        self.properties_df = pd.read_csv(properties_dir)

        self.sensors_c_index = 0
        self.drones_c_index = 0
        self.properties_c_index = 0

        self.updating_sensors = False
        self.updating_drones = False
        self.updating_properties = False

        self.online_db = MQTTHandler(self.incoming_data)

    def needs_update(self):
        needs_ud = []
        if self.sensors_c_index < self.sensors_df.shape[0] and not self.updating_sensors:
            needs_ud.append('sensors')
        if self.drones_c_index < self.drones_df.shape[0] and not self.updating_sensors:
            needs_ud.append('drones')
        if self.properties_c_index < self.properties_df.shape[0] and not self.updating_sensors:
            needs_ud.append('properties')
        return needs_ud

    def incoming_data(self, new_data, target):
        if target == "sensors":
            self.sensors_df = self.sensors_df.append(new_data, ignore_index=True)
        if target == "drones":
            self.drones_df = self.drones_df.append(new_data, ignore_index=True)
        if target == "properties":
            self.properties_df = self.properties_df.append(new_data, ignore_index=True)

# def _connect_mongo(host, port, username, password, db):
#     """ A util for making a connection to mongo """
#     if username and password:
#         mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
#         conn = MongoClient(mongo_uri)
#     else:
#         conn = MongoClient(host, port)
#
#     return conn[db]
#
#
# def read_mongo(db, collection, query=None, host='localhost', port=27017, username=None, password=None,
#                no_id=True):
#     """ Read from Mongo and Store into DataFrame """
#
#     # Connect to MongoDB
#     if query is None:
#         query = {}
#     db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
#
#     # Make a query to the specific DB and Collection
#     cursor = db[collection].find(query)
#
#     # Expand the cursor and construct the DataFrame
#     df = pd.DataFrame(list(cursor))
#
#     # Delete the _id
#     print(df)
#     if no_id:
#         del df['_id']
#     print(df)
#
#     return df
