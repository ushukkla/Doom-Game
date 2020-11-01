def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val

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

        x_t = game_state.screen_buffer # 480 x 640
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.stack(([x_t]*4), axis=2)
        s_t = np.expand_dims(s_t, axis=0)

        life = 0 #starting the game initially with 0 life, and gradually this list changes and we can see while the bot dies

        while not game.is_episode_finished():

            a_t = np.zeros([action_size])
            r_t = 0 
      

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

            agent.append_sample(s_t, action_idx, r_t)
            prev_misc = misc
            t += 1
           

            if (is_terminated and t > agent.observe):
                loss = agent.train_model()

            if t % 10000 == 0:
                print("Save model")
                agent.save_model("E:/Masters/Semester 2/VizDoom-Keras-RL-master/VizDoom-Keras-RL-master/Models/a2c")

            state = ""
            if t >= agent.observe:
                state = "train mode"
            else:
                state = "Observe mode"

            if (is_terminated):
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
