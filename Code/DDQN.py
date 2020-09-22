def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val

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
