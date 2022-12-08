import csv
from os import path, makedirs

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


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


# Source for crop, resize, rgb2yuv & preprocess functions
# https://github.com/llSourcell/How_to_simulate_a_self_driving_car
def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def data_array(list: list):
    images = []
    measurements = []
    for line in list:
        img = preprocess(cv2.imread(f"{data_dir}/{line[0]}"))
        steering = line[1]  # steering

        # Add original img
        images.append(img)
        measurements.append(steering)

        # Flip image and steering angles and add to list
        images.append(cv2.flip(img, 1))
        # steering, throttle, reverse, speed = measurement
        measurements.append(-steering)

    images = np.array(images)
    labels = np.array(measurements)
    return images, labels


def data_split(X, y):
    # now we can split the data into a training (80), testing(20), and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid


def build_model():
    # define the model
    # initializing the CNN
    model = Sequential()

    # Normalizes incoming inputs. First layer needs the input shape to work
    # model.add(BatchNormalization())

    # add model layers
    # Image normalization to avoid saturation and make gradients work better.
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
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

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=["mean_squared_error"])

    return model


def get_callbacks():
    # Automatically stop training the model when the validation loss doesn't decrease more than '0' over a period of '10' epochs
    callback_early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
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
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model accuracy')
    plt.ylabel('mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# Only run when this script is called directly
if __name__ == "__main__":
    lines = read_driver_log(data_dir)
    print(f"Read {len(lines)} lines")
    train_images, labels = data_array(lines)

    # Shuffle images along with their labels, then split into training/validation sets
    train_images, labels = shuffle(train_images, labels)
    X_train, X_val, y_train, y_val = data_split(train_images, labels)
    print(f"Created {len(X_train)} samples")

    # Using a generator to help the model use less data
    # Channel shifts help with shadows slightly
    datagen = ImageDataGenerator(channel_shift_range=0.2)
    # datagen.fit(X_train)

    model = build_model()

    # with tf.device('/cpu:0'):  # Run on the CPU to decrease the chance of it crashing do to running out of video memory.
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=500,
        callbacks=get_callbacks(),
        validation_data=(X_val, y_val))

    plotLosses(history)
    plotAccuracy(history)

    # save the model
    model.save('model/model.h5')
