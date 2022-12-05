import csv
from os import path, makedirs

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential


# Try to fix the "ran out of memory trying to allocate" error
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# Function to check if a directory exists, if not, make this directory
def check_dir(directory: str):
    dir_exists = path.isdir(directory)
    if not dir_exists:
        makedirs(directory)
    return directory


def relative_img_path(full_path: str):
    relative_path = "/".join(full_path.split('\\')[-2:])
    return relative_path


data_dir = '../Recording Lake'
batch_size = 32
target_size = (160, 320)


def read_driver_log(path: str):
    csv_path = path + '/driving_log.csv'
    # drive_df = pd.read_csv(csv_path, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    lines = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        #    0         1       2         3            4          5         6
        # 'center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'
        for line in reader:
            center_img = relative_img_path(line[0])
            left_img = relative_img_path(line[1])
            right_img = relative_img_path(line[2])
            steering = float(line[3])
            throttle = float(line[4])
            reverse = float(line[5])
            speed = float(line[6])
            # Center img
            lines.append([center_img, steering, throttle, reverse, speed])
            # Left img
            lines.append([left_img, steering + 0.2, throttle, reverse, speed])
            # Right img
            lines.append([right_img, steering - 0.2, throttle, reverse, speed])
    return lines


def data_array(list: list):
    images = []
    measurements = []
    for line in list:
        img = cv2.imread(f"{data_dir}/{line[0]}")
        # measurement = [*line[1:]]  # steering, throttle, reverse, speed
        measurement = line[1]  # steering

        # Add original img
        images.append(img)
        measurements.append(measurement)

        # Flip image and steering angles and add to list
        images.append(cv2.flip(img, 1))
        # steering, throttle, reverse, speed = measurement
        # measurements.append([-steering, throttle, reverse, speed])
        measurements.append(-measurement)

    x = np.array(images)
    y = np.array(measurements)
    return x, y


def build_model():
    # define the model
    # initializing the CNN
    model = Sequential()

    # add model layers
    model.add(Conv2D(24, (5, 5), input_shape=(target_size[0], target_size[1], 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])

    return model


def get_callbacks():
    # Automatically stop training the model when the validation loss doesn't decrease more than '0' over a period of '20' epochs
    callback_early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        mode='min')

    return [callback_early_stopping]


def plotLosses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plotAccuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    lines = read_driver_log(data_dir)
    print(f"Read {len(lines)} lines")
    X_train, y_train = data_array(lines)
    print(f"Created {len(X_train)} samples")

    model = build_model()

    with tf.device('/cpu:0'):  # Run on the CPU to decrease the chance of it crashing do to running out of video memory.
        history = model.fit(
            X_train,
            y_train,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=500,
            callbacks=get_callbacks(),
            validation_split=0.20)

    plotLosses(history)
    plotAccuracy(history)

    # save the model
    model.save('model/model.h5')
