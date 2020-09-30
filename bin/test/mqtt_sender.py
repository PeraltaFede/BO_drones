import paho.mqtt.client as mqtt

client = mqtt.Client("sender")
client.connect('127.0.0.1', 1883, 60)
while True:
    a = input('lol')
    if a == "1":
        client.publish("drone",
                       '{"type": "drone", "armed":true, "id": 4, "online": true, "position": {"lat": -25.320924, "lon": -57.34070218, "alt": 100.093, "yaw": 2.345768690109253}, "goal": {"lat": -25.329965691489356, "lon": -57.347274975, "alt": 100.051, "yaw": 0}, "velocity": [-1.43, 1.39, 0.0], "energy_left": 1, "pri_pu_states": [0, 3], "timestamp": 1575753886.9177554}')
    else:
        client.publish("drone",
                       '{"type": "drone", "armed":false ,"id": 4, "online": false, "position": {"lat": -25.329724, "lon": -57.3475218, "alt": 100.093, "yaw": 2.345768690109253}, "goal": {"lat": -25.329965691489356, "lon": -57.347274975, "alt": 100.051, "yaw": 0}, "velocity": [0.5, 0.50, 0.0], "energy_left": 0.5, "pri_pu_states": [0, 3], "timestamp": 1575753886.9177554}')
#
# ids = ["type", "longitude", "latitude", "value", "altitude", "drone", "timestamp"]
# while True:
#     msg = "t," + input("type x, y, value for t: ") + ",1, 2, 3"
#     payload = dict(zip(ids, msg.split(",")))
#     for key in payload:
#         try:
#             payload[key] = float(payload[key])
#         except Exception as e:
#             pass
#     client.publish("sensors", str(payload))
#     print("Message {} sent".format(payload))
