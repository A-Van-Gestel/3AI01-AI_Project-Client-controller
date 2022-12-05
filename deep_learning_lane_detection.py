import csv

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator

from os import path, makedirs


# Function to check if a directory exists, if not, make this directory
def check_dir(directory: str):
    dir_exists = path.isdir(directory)
    if not dir_exists:
        makedirs(directory)
    return directory


def relative_img_path(full_path: str):
    relative_path = full_path.split('\\')[-1]
    return relative_path


data_dir = '../Recording Lake'
batch_size = 64
target_size = (320, 160)


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


if __name__ == "__main__":
    lines = read_driver_log(data_dir)
    print(lines)
