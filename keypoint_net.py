from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.backend import tensorflow_backend as tb


class KeyPointNet:

    def __init__(self, input_shape, num_class, gpu=True):
        self.input_shape = input_shape
        self.num_class = num_class

        if gpu == True:
            self.device = '/gpu:0'
        elif gpu == False:
            self.device = '/cpu:0'

    def build(self):
        with tb.tf.device(self.device):
            model = Sequential()
            # input layer
            model.add(BatchNormalization(input_shape=self.input_shape))
            model.add(Conv2D(24, (5, 5), kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.2))
            # layer 2
            model.add(Conv2D(36, (5, 5)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.2))
            # layer 3
            model.add(Conv2D(48, (5, 5)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.2))
            # layer 4
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.2))
            # layer 5
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(Flatten())
            # layer 6
            model.add(Dense(500, activation="relu"))
            # layer 7
            model.add(Dense(90, activation="relu"))
            # layer 8
            model.add(Dense(self.num_class))

            return model
