import csv
import math
from os import path, makedirs

import cv2
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
from keras import callbacks
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

model_name = 'model_augmentation_all'


# Function to check if a directory exists, if not, make this directory
def check_dir(directory: str):
    dir_exists = path.isdir(directory)
    if not dir_exists:
        makedirs(directory)
    return directory


def relative_img_path(full_path: str):
    relative_path = "/".join(full_path.split('\\')[-2:])
    return relative_path


data_dirs = [
    '../Recording Jungle [Left lane]',
    '../Recording Jungle [Right lane]',
    '../Recording Lake',
    '../Recording Mountain [Left lane]',
    '../Recording Mountain [Right lane]'
]
batch_size = 32
target_size = (160, 320)


def read_driver_log(paths: list):
    lines = []
    for path in paths:
        csv_path = path + '/driving_log.csv'

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
                lines.append([center_img, steering, throttle, reverse, speed, path])
                # Left img
                lines.append([left_img, steering + 0.2, throttle, reverse, speed, path])
                # Right img
                lines.append([right_img, steering - 0.2, throttle, reverse, speed, path])
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


def preprocess(image, conv_yuv=False):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    if conv_yuv:
        image = rgb2yuv(image)
    return image


# Source for generate_shadow_coordinates & add_shadow functions
# https://www.freecodecamp.org/news/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f/
def generate_shadow_coordinates(img_shape, no_of_shadows=1):
    vertices_list = []
    for index in range(no_of_shadows):
        vertex = []
        for dimensions in range(np.random.randint(3, 15)):  # Dimensionality of the shadow polygon
            vertex.append((img_shape[1] * np.random.uniform(), img_shape[0] // 3 + img_shape[0] * np.random.uniform()))
            vertices = np.array([vertex], dtype=np.int32)  # single shadow vertices
            vertices_list.append(vertices)
            return vertices_list  # List of shadow vertices


def add_shadow(image, no_of_shadows=1):
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Conversion to HLS
    mask = np.zeros_like(image)
    img_shape = image.shape
    vertices_list = generate_shadow_coordinates(img_shape, no_of_shadows)  # 3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices,
                     255)  # adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_hls[:, :, 1][mask[:, :, 0] == 255] = image_hls[:, :, 1][mask[:, :,
                                                                  0] == 255] * 0.5  # if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)  # Conversion to RGB
    return image_rgb


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_flip(image, steering_angle):
    if np.random.rand() < 0.6:
        # Flip image and steering angles
        image = (cv2.flip(image, 1))
        steering_angle = -steering_angle
    return image, steering_angle


def augment(image, steering_angle):
    image, steering_angle = random_flip(image, steering_angle)
    image = random_brightness(image)
    image = add_shadow(image, np.random.randint(1, 3))
    return image, steering_angle


def data_generation(lines, batch_size, is_training=True):
    num_lines = len(lines)

    while True:
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset + batch_size]

            images = []
            measurements = []

            for line in batch_lines:
                path = line[-1]
                img = preprocess(cv2.imread(f"{path}/{line[0]}"))
                steering = line[1]  # steering

                # argumentation if training
                if is_training and np.random.rand() < 0.6:
                    img, steering = augment(img, steering)

                img = rgb2yuv(img)
                images.append(img)
                measurements.append(steering)

            yield shuffle(np.array(images), np.array(measurements))


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
    plt.savefig(f'{check_dir("plots")}/{model_name}.png')


# Only run when this script is called directly
if __name__ == "__main__":
    lines = read_driver_log(data_dirs)
    print(f"Read {len(lines)} lines")

    # Shuffle images along with their labels, then split into training/validation sets
    train_dataset, val_dataset = train_test_split(lines, test_size=0.2, random_state=0)
    print(f"Created {len(train_dataset)} train samples")
    print(f"Created {len(val_dataset)} validation samples")

    # Using a generator to help the model use less data
    training_generator = data_generation(train_dataset, batch_size=batch_size)
    validation_generator = data_generation(val_dataset, batch_size=batch_size, is_training=False)

    steps_per_epoch = int(math.ceil(len(train_dataset) / batch_size))
    validation_steps = int(math.ceil(len(val_dataset) / batch_size))

    model = build_model()

    # with tf.device('/cpu:0'):  # Run on the CPU to decrease the chance of it crashing do to running out of video memory.
    history = model.fit(
        training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        callbacks=get_callbacks(),
        validation_data=validation_generator,
        validation_steps=validation_steps)

    plotLosses(history)

    # save the model
    model.save(f'{check_dir("model")}/{model_name}.h5')
