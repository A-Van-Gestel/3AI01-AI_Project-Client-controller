import argparse  # parsing command line arguments
import shutil  # high level file operations
import base64  # decoding camera images
import os  # write + read files
import random
from datetime import datetime  # to set frame timestamp
from io import BytesIO  # manipulate string and byte data in memory

import cv2
import eventlet.wsgi  # web server gateway interface
import numpy as np  # matrix math
import socketio  # real-time server
from PIL import Image  # image manipulation
from flask import Flask  # framework for web devices

from opencv_lane_detection import process_image

# from keras.models import load_model  # load our saved model

# Set resulting image width & height
height = 320
width = 160

# set min/max speed for our autonomous car
max_speed = 30
min_speed = 10

# and a speed limit
speed_limit = max_speed


def resize(image):
    return cv2.resize(image, (width, height), cv2.INTER_AREA)


# initialize our server
sio = socketio.Server(always_connect=True)
# flask web app
application = Flask(__name__)

# init our model and image array as empty
net = None
image_array_before = None


# Server event handler
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Read data Unity sends us
        steering_angle = float(data["steering_angle"].replace(",", "."))
        throttle = float(data["throttle"].replace(",", "."))
        speed = float(data["speed"].replace(",", "."))
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        try:
            image = np.asarray(image)
            # image = resize(image)
            # image = np.array([image])

            # steering_angle = float(net.predict(image))
            # steering_angle = random.uniform(-25.0, 25.0)
            steering_angle = process_image(image)
            # steering_angle = 1.0

            global speed_limit
            if speed > speed_limit:
                speed_limit = min_speed
            else:
                speed_limit = max_speed
            throttle = (1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2)
            # throttle = random.uniform(0.0, 10.0)

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
    steering_angle /= 25
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
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='model.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # load model
    # model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)  # Could be dangerous
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    application = socketio.Middleware(sio, application)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('localhost', 4567)), application)
