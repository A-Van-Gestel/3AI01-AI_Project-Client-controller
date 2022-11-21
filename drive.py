import base64  # for lossless encoding transfer
import os  # write + read files
import random
from datetime import datetime  # to set frame timestamp
from io import BytesIO  # manipulate string and byte data in memory

import cv2
import eventlet.wsgi
import numpy as np
import socketio  # server
from PIL import Image
from flask import Flask  # framework for web devices
from keras.models import load_model

height = 320
width = 160


def resize(image):
    return cv2.resize(image, (width, height), cv2.INTER_AREA)


# server init
sio = socketio.Server(always_connect=True)
# flask web app
application = Flask(__name__)

# init empty model and image array
net = None
image_array_before = None

# Speed limits
max_speed = 30
min_speed = 10

speed_limit = max_speed


# Server event handler
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"].replace(",", "."))
        throttle = float(data["throttle"].replace(",", "."))
        speed = float(data["speed"].replace(",", "."))
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # save frame
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(r'./path', timestamp)
        image.save('{}.jpg'.format(image_filename))

        try:
            image = np.asarray(image)
            image = resize(image)
            image = np.array([image])

            # steering_angle = float(net.predict(image))
            steering_angle = random.uniform(-25.0, 25.0)

            global speed_limit
            if speed > speed_limit:
                speed_limit = min_speed
            else:
                speed_limit = max_speed
            # throttle = (1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2)
            throttle = random.uniform(0.0, 10.0)

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)

    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    steering_angle = str(steering_angle).replace(".", ",")
    throttle = str(throttle).replace(".", ",")

    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle,
            "throttle": throttle
        },
        skip_sid=True)


if __name__ == "__main__":
    # net = load_model('path')
    application = socketio.Middleware(sio, application)
    # deploy
    eventlet.wsgi.server(eventlet.listen(('localhost', 4567)), application)
