#!/d/Anaconda/envs/gpu/Scripts/python
from __future__ import print_function
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
from keras.models import model_from_json
from vizdoom import DoomGame, ScreenResolution
from networks import Networks
from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda,  Activation, Embedding
import time
import matplotlib.pyplot as plt
import json
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf
import skimage as skimage
from skimage import transform, color, exposure

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val

    
class REINFORCEAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.observe = 0
        self.model = None
        self.states=[]
        self.actions=[] 
        self.rewards = []
        self.stats_window_size= 50 
        self.frame_per_action = 4 
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.nb_epoch = 5
        self.mavg_score = []
        self.var_score = [] 
        self.mavg_ammo_left = [] 
        self.latent_dim = 2
        self.value_size = 1
        self.observe = 0
        self.frame_per_action = 4

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_action(self, state):
        policy = self.model.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def shape_reward(self, r_t, misc, prev_misc, t):
        
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1
        elif (misc[2] < prev_misc[2]): 
            r_t = r_t - 0.1
        elif (misc[1] < prev_misc[1]): 
            r_t = r_t - 0.1
        return r_t

    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards) 
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            print ('std = 0!')
            return 0

        update_inputs = np.zeros(((episode_length,) + self.state_size)) 
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i,:,:,:] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]
        
        loss = self.model.fit(update_inputs, advantages, nb_epoch=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

        return loss.history['loss']


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3) 
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img) 

    return img


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("E:/Masters/Semester 2/VizDoom-Keras-RL-master/VizDoom-Keras-RL-master/scenarios/defend_the_center.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    max_episodes = 20000

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables 
    prev_misc = misc
    img_rows , img_cols = 64, 64
    img_channels = 4 

    action_size = game.get_available_buttons_size()
    state_size = (img_rows, img_cols, img_channels)
    agent = REINFORCEAgent(state_size, action_size)

    agent.model = Networks.policy_reinforce(state_size, action_size, agent.learning_rate)
    life_buffer=[]
    ammo_buffer=[]
    kills_buffer=[] 
    loss=[0]

    GAME = 0
    max_life = 0 
    t = 0

    avg_loss=[]
    for i in range(max_episodes):

        game.new_episode()
        game_state = game.get_state()
        misc = game_state.game_variables 
        prev_misc = misc

        life = 0 
        x_t = game_state.screen_buffer 
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.stack(([x_t]*4), axis=2) 
        s_t = np.expand_dims(s_t, axis=0) 

        while not game.is_episode_finished():
            r_t = 0 
            a_t = np.zeros([action_size]) 

            x_t = game_state.screen_buffer
            x_t = preprocessImg(x_t, size=(img_rows, img_cols))
            x_t = np.reshape(x_t, (1, img_rows, img_cols, 1))
            s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

            action_idx, policy  = agent.get_action(s_t)
            a_t[action_idx] = 1 

            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action
            game.advance_action(skiprate)

            r_t = game.get_last_reward()  
            is_terminated = game.is_episode_finished()

            if (is_terminated):
                if (life > max_life):
                    max_life = life 
                life_buffer.append(life)
                ammo_buffer.append(misc[1])
                kills_buffer.append(misc[0])
                avg_loss.append(loss[0])
                print ("Finishing episode ", prev_misc, policy)
            else:
                life += 1
                game_state = game.get_state()  
                misc = game_state.game_variables

            r_t = agent.shape_reward(r_t, misc, prev_misc, t)

            agent.append_sample(s_t, action_idx, r_t)

            t += 1
            prev_misc = misc

            if (is_terminated and t > agent.observe):
                loss = agent.train_model()

            if t % 10000 == 0:
                print("Saving the model")
                agent.model.save_weights("E:/Masters/Semester 2/VizDoom-Keras-RL-master/VizDoom-Keras-RL-master/Models/reinforce.h5", overwrite=True)

            if (is_terminated):

                print("/ Episode", GAME, "/Action", action_idx, "/ Reward", r_t,  "/ Loss", loss)

        GAME += 1


plt.figure(1)             
plt.plot(avg_loss)
plt.title('Loss vs Episodes')
plt.ylabel('Loss')
plt.xlabel('Episodes')

plt.figure(2)             
plt.plot(kills_buffer)
plt.ylabel('Average Kill Counts')
plt.xlabel('Episodes')