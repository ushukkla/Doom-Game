#!/d/Anaconda/envs/gpu/Scripts/python
from __future__ import print_function
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
from vizdoom import DoomGame, ScreenResolution
import matplotlib.pyplot as plt
import json
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K
from vizdoom import *
import itertools as it
import tensorflow as tf
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10 ** 4)
        self.training_frames = 10 ** 7
        self.image_sequence_size = 4
        self.frameskip = 4
        self.image_sequence = deque(maxlen=4)
        self.save_path = "save/"
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 10 ** 5
        self.learning_rate = 0.00025
        try:
            self.load()
        except FileNotFoundError:
            self.model = self._build_model()
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs
class A2CAgent:

    def __init__(self, state_size, action_size):
      
        self.state_size = state_size
        self.action_size = action_size
        self.frame_per_action = 4
        self.var_score = []
        self.mavg_ammo_left, self.mavg_score  = [], []
        self.mavg_kill_counts = [] 
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.states= []
        self.actions =[]
        self.rewards = []
        self.actor = None
        self.value_size = 1
        self.observe = 0
        self.critic = None
        self.stats_window_size= 50 

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_action(self, state):
        policy = self.actor.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy
    
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)

        discounted_rewards -= np.mean(discounted_rewards) 
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.state=[]
            self.actions=[] 
            self.rewards=[]
            print ('std = 0!')
            return 0

        update_inputs = np.zeros(((episode_length,) + self.state_size)) 
        for i in range(episode_length):
            update_inputs[i,:,:,:] = self.states[i]


        values = self.critic.predict(update_inputs)

        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]
            
        actor_loss = self.actor.fit(update_inputs, advantages, nb_epoch=1, verbose=0)
        critic_loss = self.critic.fit(update_inputs, discounted_rewards, nb_epoch=1, verbose=0)

        self.states=[] 
        self.actions=[] 
        self.rewards =[]

        return actor_loss.history['loss'], critic_loss.history['loss']

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5", overwrite=True)
        self.critic.save_weights(name + "_critic.h5", overwrite=True)


    def shape_reward(self, r_t, misc, prev_misc, t):
        
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        elif (misc[2] < prev_misc[2]): 
            r_t = r_t - 0.1

        elif (misc[1] < prev_misc[1]): 
            r_t = r_t - 0.1

        return r_t

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5", overwrite=True)
        self.critic.load_weights(name + "_critic.h5", overwrite=True)
 	def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
def __len__(self):
        return self.nenvs

def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3) # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img) 

    return img

if __name__ == "__main__":


    # Avoid Tensorflow eats up GPU memory
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
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()
    img_channels = 4 
    img_rows , img_cols = 64, 64



    state_size = (img_rows, img_cols, img_channels)
    agent = A2CAgent(state_size, action_size)
    agent.actor = Networks.actor_network(state_size, action_size, agent.actor_lr)
    agent.critic = Networks.critic_network(state_size, agent.value_size, agent.critic_lr)

    GAME = 0
    t = 0
    max_life = 0 
    life_buffer=[]
    ammo_buffer=[]
    kills_buffer=[]
    loss=[0]
    avg_loss=[]

    for i in range(max_episodes):

        game.new_episode()
        game_state = game.get_state()
        misc = game_state.game_variables 
        prev_misc = misc

        size_i = game_state.screen_buffer # 480 x 640
        size_i = preprocessImg(size_i, size=(img_rows, img_cols))
        size_j = np.stack(([size_i]*4), axis=2)
        size_j = np.expand_dims(size_j, axis=0)

        life = 0 

        while not game.is_episode_finished():

            a_t = np.zeros([action_size])
            r_t = 0 
      

            size_i = game_state.screen_buffer
            size_i = preprocessImg(size_i, size=(img_rows, img_cols))
            size_i = np.reshape(size_i, (1, img_rows, img_cols, 1))
            size_j = np.append(size_i, size_j[:, :, :, :3], axis=3)
                
    
            action_idx, policy  = agent.get_action(size_j)
            a_t[action_idx] = 1 

            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action 
            game.advance_action(skiprate)

            r_t = game.get_last_reward()  
            isize_jerminated = game.is_episode_finished()

            if (isize_jerminated):
                # Save max_life
                if (life > max_life):
                    max_life = life 
                life_buffer.append(life)
                ammo_buffer.append(misc[1])
                kills_buffer.append(misc[0])

                print ("Finishing episode ", prev_misc, policy)
            else:
                life += 1
                game_state = game.get_state()  # Observe again after we take the action
                misc = game_state.game_variables

            r_t = agent.shape_reward(r_t, misc, prev_misc, t)

            agent.append_sample(size_j, action_idx, r_t)
            prev_misc = misc
            t += 1
           

            if (isize_jerminated and t > agent.observe):
                loss = agent.train_model()

            if t % 10000 == 0:
                print("Save model")
                agent.save_model("E:/Masters/Semester 2/VizDoom-Keras-RL-master/VizDoom-Keras-RL-master/Models/a2c")

            state = ""
            if t >= agent.observe:
                state = "train mode"
            else:
                state = "Observe mode"

            if (isize_jerminated):
                print("/ Episode", i, "/ ACTION", action_idx, "/ REWARD", r_t,  "/ LOSS", loss)


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