import csv
import random


import cv2
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.optimizers import Adam



def read_driving_log(filename='./data/driving_log.csv'):
    """Parse csv file of driving log and return as a python list."""
    lines = []

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header line
        for line in reader:
            lines.append(line)
    return lines


def image_generator(samples, batch_size=32):
    """Yields batches of images and steering angles as training data. Used
    by keras.fit_generator"""
    num_samples = len(samples)

    while True:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = plt.imread(name)
                center_angle = float(batch_sample[3])

                # I use the left and right cameras to add more training data
                # and for turning training data. I just add/subtract an angle
                # of 0.3 to the center steering to make it 0
                correction = 0.3
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                left_image = plt.imread(name.replace('center_', 'left_'))
                right_image = plt.imread(name.replace('center_', 'right_'))

                images.append(left_image)
                angles.append(steering_left)
                images.append(right_image)
                angles.append(steering_right)

                images.append(center_image)
                angles.append(center_angle)

                # flip image
                images.append(np.fliplr(center_image))
                angles.append(center_angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train



def create_model(input_shape):
    """Creates a convolution neural network using the Nvidia architecture. I
    also add a normalization layer and cropping layer at the beginning."""
    model = Sequential([
        Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape),
        Cropping2D(cropping=((50,20), (0,0))),
        Conv2D(24, 5, 5, subsample=(2,2), activation='relu'),
        Conv2D(36, 5, 5, subsample=(2,2), activation='relu'),
        Conv2D(48, 5, 5, subsample=(2,2), activation='relu'),
        Conv2D(64, 3, 3, activation='relu'),
        Conv2D(64, 3, 3, activation='relu'),
        Flatten(),
        Dense(100),
        Dense(50),
        Dense(10),
        Dense(1),
    ])
    return model


def train_model(input_shape=(160, 320, 3), output_file='model.h5'):
    """Main function to train the model. I train with a batch size of 32
    for 3 epochs."""
    samples = read_driving_log()

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = image_generator(train_samples, batch_size=32)
    validation_generator = image_generator(validation_samples, batch_size=32)

    model = create_model(input_shape)
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples),
                    nb_epoch=3)

    model.save(output_file)
    return model




