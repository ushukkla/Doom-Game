#!/d/Anaconda/envs/gpu/Scripts/python
from __future__ import print_function
import skimage as skimage
import json
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
import random
from random import choice
import numpy as np
from collections import deque
import time
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
import tensorflow as tf


class Networks(object):

    def dqn(input_shape, action_size, learning_rate):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(input_shape)))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=512, activation='relu'))
        model.add(Dense(output_dim=action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    def actor_network(input_shape, action_size, learning_rate):


        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=action_size, activation='softmax'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model
   
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):


        state_input = Input(shape=(input_shape)) 
        cnn_feature = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(state_input)
        cnn_feature = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(cnn_feature)
        cnn_feature = Convolution2D(64, 3, 3, activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='relu')(cnn_feature)

        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

  
    def critic_network(input_shape, value_size, learning_rate):


        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=value_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    def policy_reinforce(input_shape, action_size, learning_rate):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=action_size, activation='softmax'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

    


