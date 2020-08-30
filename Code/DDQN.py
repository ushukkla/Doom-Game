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

class DoubleDQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.var_score = []
        self.mavg_ammo_left = [] 
        self.mavg_kill_counts = [] 
        self.action_size = action_size
        self.model = None
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.target_model = None
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.observe = 500
        self.explore = 50000 
        self.frame_per_action = 4
        self.update_target_freq = 3000 
        self.timestep_per_train = 100 
        #performance measure
        self.stats_window_size= 50 
        self.mavg_score = [] 
        self.memory = deque()
        self.max_memory = 50000 
        self.epsilon_std = 0.01
        self.nb_epoch = 5
        self.latent_dim = 2
        self.intermediate_dim = 128
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def shape_reward(self, r_t, misc, prev_misc, t):

        if(misc[1] < prev_misc[1]): 
            r_t = r_t - 0.1

        elif(misc[0] > prev_misc[0]):
            r_t = r_t + 1

        elif (misc[2] < prev_misc[2]): #losing health
           r_t = r_t - 0.1

        return r_t

    def get_action(self, state):
        if np.random.rand() >= self.epsilon: #using epsioln greedy policy
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        else:
            action_idx = random.randrange(self.action_size)
        return action_idx

    def train_minibatch_replay(self):

        batch_size = min(self.batch_size, len(self.memory)) #training single batch
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros(((batch_size,) + self.state_size)) 
        action = []
        reward = []
        done = []
        update_target = np.zeros(((batch_size,) + self.state_size))

        for i in range(batch_size):
            update_input[i,:,:,:] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i,:,:,:] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input) 

        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(self.batch_size):
  
            if not done[i]:
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

            else:
                target[i][action[i]] = reward[i]

        loss = self.model.train_on_batch(update_input, target) #make some small batches

        return np.max(target[-1]), loss

    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t): #saving to the replay memory
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        elif self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        elif t % self.update_target_freq == 0:
            self.update_target_model()

    def train_replay(self): #picking the samples
        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_target = np.zeros(((num_samples,) + self.state_size))
        update_input = np.zeros(((num_samples,) + self.state_size)) 
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i,:,:,:] = replay_samples[i][0]
            reward.append(replay_samples[i][2])
            action.append(replay_samples[i][1])
            update_target[i,:,:,:] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self.model.predict(update_input) 
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)
               
        for i in range(num_samples):
            if not done[i]:
                a = np.argmax(target_val[i]) #taking updates from the target model
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])
            else:
                target[i][action[i]] = reward[i]
        
        loss = self.model.fit(update_input, target, batch_size=self.batch_size, nb_epoch=1, verbose=0)
        return (np.max(target[-1]), loss.history['loss'])

    def save_model(self, name):
        self.model.save_weights(name)

    def load_model(self, name): #loading the saved model
        self.model.load_weights(name)

    def padState(state, only_2d = False):
    # pad the inputs from the top and the bottom to from 120 to 128 to fit the network
        if only_2d:
            return np.lib.pad(state, ((6, 6), (0, 0)), 'constant', constant_values=(0))
        else:
            return np.lib.pad(state, ((0, 0), (6, 6), (0, 0)), 'constant', constant_values=(0))

def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img

if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    game = DoomGame()
    #calling the scenario file
    game.load_config("E:/Masters/Semester 2/VizDoom-Keras-RL-master/VizDoom-Keras-RL-master/scenarios/defend_the_center.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()
    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables 
    prev_misc = misc
    img_rows , img_cols = 64, 64
    img_channels = 4 
    action_size = game.get_available_buttons_size()
    state_size = (img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, action_size)
    agent.model = Networks.dqn(state_size, action_size, agent.learning_rate)
    agent.target_model = Networks.dqn(state_size, action_size, agent.learning_rate)
    #training variables
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0 
    avg_loss=[]
    life = 0  
    avg_rew=[]
    life_buffer=[] 
    ammo_buffer=[] 
    kills_buffer=[]  #buffer to computing the numbers
    x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4
    is_terminated = game.is_episode_finished()

    r_t = 0
    loss= [0]
    Q_max = 0

    for i in range(20000):

        while not game.is_episode_finished():
            a_t = np.zeros([action_size])
            action_idx  = agent.get_action(s_t)
            a_t[action_idx] = 1
    
            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action
            game.advance_action(skiprate)
            game_state = game.get_state()  
            is_terminated = game.is_episode_finished()
            r_t = game.get_last_reward()  
    
            if (is_terminated):
                if (life > max_life):
                    max_life = life
                GAME += 1
                i+=1
                life_buffer.append(life)
                ammo_buffer.append(misc[1])
                kills_buffer.append(misc[0])
                avg_loss.append(loss[0])
                print ("Episode Finished", misc)
                game.new_episode()
                game_state = game.get_state()
                x_t1 = game_state.screen_buffer
                misc = game_state.game_variables
                print("Episode :", i, "/ Epsilon: ", agent.epsilon, "/ Action: ", action_idx, "/ Reward: ", r_t,   "/ Loss: ", loss[0])
                break
    
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer
            x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
            x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
            r_t = agent.shape_reward(r_t, misc, prev_misc, t)
            prev_misc = misc #updating the cache
    
            if (not is_terminated):
                life += 1
            else:
                life = 0
            # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
            agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)
    
            # Do the training
            if t % agent.timestep_per_train == 0:
                Q_max, loss = agent.train_replay()

            t += 1   
            s_t = s_t1
            if t % 10000 == 0: #progress saving with 10k iterations
                agent.model.save_weights("E:/Masters/Semester 2/VizDoom-Keras-RL-master/VizDoom-Keras-RL-master/Models/ddqn.h5", overwrite=True)
                print("Model saved successfully!!!")
       
plt.figure(1)             
plt.plot(avg_loss)
plt.title('Loss vs Episodes')
plt.ylabel('Loss')
plt.xlabel('Episodes')

plt.figure(2)             
plt.plot(kills_buffer)
plt.ylabel('Average Kill Counts')
plt.xlabel('Episodes')