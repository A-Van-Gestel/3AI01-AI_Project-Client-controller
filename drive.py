import argparse  # parsing command line arguments
import base64  # decoding camera images
import os  # write + read files
import shutil  # high level file operations
from datetime import datetime  # to set frame timestamp
from io import BytesIO  # manipulate string and byte data in memory

import cv2
import eventlet.wsgi  # web server gateway interface
import numpy as np  # matrix math
import socketio  # real-time server
from PIL import Image  # image manipulation
from flask import Flask  # framework for web devices
from keras.models import load_model  # load our saved model

# from opencv_lane_detection import process_image
from road_lane_detection import process_image

from deep_learning_lane_detection import preprocess

# --- Settings ---
prediction_mode = 'cnn'
# model_name = 'model_augmentation_test'
model_name = 'model_augmentation_lake_only'
model_path = f'model/{model_name}.h5'

# Target speed for our autonomous car, it will always try to keep this speed
target_speed = 15

# Max steering angle for our autonomous car
max_steering_angle = 25

# steering angle smoothing options (Only in OpenCV mode)
angle_smoothing = True
smoothing_strength = 10
angle_history = [0.0] * smoothing_strength  # Fill at startup fully with 0.0 based on smoothing strength


# --- Application ---
# initialize our server
sio = socketio.Server(always_connect=True)
# flask web app
application = Flask(__name__)

# init our model as empty
model = None


# Server event handler
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Read data Unity sends us
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        try:
            image = np.asarray(image)

            if prediction_mode == 'cnn':
                image = preprocess(image, conv_yuv=True)  # apply the preprocessing
                image = np.array([image])  # the model expects 4D array
                steering_angle = float(model.predict(image))
            else:
                steering_angle = process_image(image)
                # Clamp steering angle between -25 & 25
                steering_angle = max_steering_angle if steering_angle > max_steering_angle else steering_angle
                steering_angle = -max_steering_angle if steering_angle < - max_steering_angle else steering_angle
                # Steering angle needs to be normalised between -1 & 1
                steering_angle /= max_steering_angle

            # Only allow steering angle smoothing in opencv mode
            if angle_smoothing and prediction_mode != 'cnn':
                # Save steering angle to history
                angle_history.pop(0)
                angle_history.append(steering_angle)

                # Get average of steering angle over the history to smooth the output
                steering_angle = sum(angle_history) / len(angle_history)
                print(f"angle_history: {angle_history}")

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            if speed > target_speed:
                throttle = -1.0  # slow down
            else:
                throttle = 1.25 - abs(
                    steering_angle)  # speed up proportional to the steering angle (eg: sharp turn = slower acceleration)

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
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle)
        },
        skip_sid=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'prediction_mode',
        type=str,
        nargs='?',
        default='cnn',
        help='opencv | cnn'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    prediction_mode = args.prediction_mode

    # Load model only if needed
    if prediction_mode == 'cnn':
        model = load_model(model_path)

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
