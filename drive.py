import argparse  # parsing command line arguments
import base64  # decoding camera images
import math
import os  # write + read files
import shutil  # high level file operations
from datetime import datetime  # to set frame timestamp
from io import BytesIO  # manipulate string and byte data in memory

import eventlet.wsgi  # web server gateway interface
import numpy as np  # matrix math
import socketio  # real-time server
from PIL import Image  # image manipulation
from flask import Flask  # framework for web devices
from keras.models import load_model  # load our saved model

from deep_learning_lane_detection import preprocess
from road_lane_detection import process_image

# --- Settings ---
prediction_mode = 'cnn'
# model_name = 'model_augmentation_lake_only_corrections'
model_name = 'model_augmentation_jungle_only_corrections'
model_path = f'model/{model_name}.h5'

# Target speed for our autonomous car, it will always try to keep this speed
max_speed = 25
min_speed = 10
target_speed = max_speed

# Max steering angle for our autonomous car (values from Unity project)
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
                # print(f"angle_history: {angle_history}")

            # Dynamically set the throttle based on the current speed, and driving angle
            global target_speed
            if abs(steering_angle * max_steering_angle) < 7.0:
                # Slow down a little on lower steering angles
                target_speed_multiplier = 1 - abs(math.log10(1 - (abs(steering_angle * max_steering_angle) - max_steering_angle/10) / max_steering_angle))
            else:
                # Slow down more on higher steering angles
                target_speed_multiplier = 1 - abs(steering_angle)
            target_speed_proposed = max_speed * target_speed_multiplier
            target_speed = target_speed_proposed if target_speed_proposed > min_speed else min_speed  # Make sure to never go slower than min speed
            print(f"target_speed: {target_speed}")

            if speed > target_speed:
                throttle = 1.25 - abs(speed / target_speed)  # slow down based on how much we're speeding
            else:
                # print(f"speed = {speed}")
                throttle_multiplier = 1 - abs(math.log10(1 - (speed - target_speed/10) / target_speed))  # decrease throttle a lot when reaching target speed
                # print(f"throttle_multiplier = {throttle_multiplier}")
                steering_angle_multiplier = 1.25 - abs(steering_angle)  # speed up proportional to the steering angle (eg: sharp turn = slower acceleration)
                throttle = throttle_multiplier * steering_angle_multiplier

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
        '-speed',
        type=float,
        nargs='?',
        default=0.0,
        help='The maximum speed the car should drive'
    )
    parser.add_argument(
        '-model',
        type=str,
        nargs='?',
        default='',
        help='Name of the model in the model folder to use in the cnn network (without .h5 extension)'
    )
    parser.add_argument(
        '-image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    prediction_mode = args.prediction_mode
    speed_args = args.speed
    model_args = args.model

    print(f"--- PREDICTION MODE: {prediction_mode} ---")

    # Set max_speed when specified
    if speed_args != 0.0:
        max_speed = speed_args
        target_speed = max_speed
        print(f"--- MAX SPEED: {target_speed} ---")

    # Load model only if needed
    if prediction_mode == 'cnn':
        # Set model if specified
        if model_args != '':
            model_name = model_args
            model_path = f'model/{model_name}.h5'
            print(f"--- LOADED MODEL: {model_path} ---")

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
