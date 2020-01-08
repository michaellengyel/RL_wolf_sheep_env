# Reinforcement algorithm structure based on:
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb

from environment import *

import tensorflow as tf
import numpy as np

from skimage import transform           # Help us to preprocess the frames
from skimage.color import rgb2gray      # Help us to gray our frames

import matplotlib.pyplot as plt

from collections import deque

import random

import warnings

warnings.filterwarnings('ignore')


def preprocess_frame(frame):
    # Grayscale frame
    gray = rgb2gray(frame)

    # Crop the screen
    # [Up:Down, Left:Right]
    cropped_frame = gray[:, :]

    # Normalized Pixel Values
    normalized_frame = cropped_frame/255.0

    # Resize
    preprocess_frame = transform.resize(normalized_frame, [15, 15])

    # Return 15x15x1
    return preprocess_frame


def stack_framesX(stacked_frames, state, is_new_episode, stack_size):
    # Preproccess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear out stacked_frames
        stacked_frames = deque([np.zeros((15, 15), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we are in a new episode, copy the same frame x4
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


'''
This function will do the part
With epsilon epsilon select a random action atat, otherwise select at=argmaxaQ(st, a)
'''


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess, dqnetwork):
    ## EPSILON GREEDY STRATAGY
    # Choose action a from state s using epsilon greedy
    exp_exp_tradeoff = np.random.rand()

    # Here we"ll use an improved version of out epsilon greedy stratagy used in Q-Learning notebook
    # Here we"ll use an improved version of out epsilon greedy stratagy used in Q-Learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_probability > exp_exp_tradeoff:
        # Make a random action (exploration)
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]

    else:
        # Get action from Q-network (exploitation)
        Qs = sess.run(dqnetwork.output, feed_dict={dqnetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability


def onehot_to_int(action):
    return np.argmax(action, axis=0)


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='dqnetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each element of a state_size in tuple

            # [None, 15, 15, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions")

            # Remember that target_q is the R(s, a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            '''
            First convnet:
            CNN
            ELU
            '''
            # Input is 15x15x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=8,
                                          kernel_size=[4, 4],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            '''
            Second convnet:
            CNN
            ELU
            '''
            # Input is 15x15x4
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=16,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            '''
            Third convnet:
            CNN
            ELU
            '''
            # Input is 15x15x4=900
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=16,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=70,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

            # Q is out predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The Loss is the difference between out predicted Q_values and Q_target
            # Sum (Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]


#def stacked_framesX(stacked_frames, next_state, param, stack_size):
#    pass


def main():
    # Env size is 15x15x3
    env = Environment("data/reduced_height_map.jpg", 15, 100, True)

    # Hardcoded action space: 8
    action_space = 8

    # One-Hot encoded actions
    possible_actions = np.array(np.identity(action_space, dtype=int)).tolist()

    # Number of frames we stack
    stack_size = 4

    # Init Deque with zero-images one array for each image
    stacked_frames = deque([np.zeros((15, 15), dtype=np.int) for i in range(stack_size)], maxlen=4)

    ### MODEL HYPERPARAMETERS
    state_size = [15, 15, 4]
    action_size = 8
    learning_rate = 0.00025

    ### TRAINING HYPERPARAMETERS
    total_episodes = 50
    max_steps = 30000
    batch_size = 64

    # Exploration parameters for epsilon greedy stratagy
    explore_start = 1.0
    explore_stop = 0.01
    decay_rate = 0.00001

    # Q Learning hyperparameters
    gamma = 0.9

    ### MEMORY HYPERPARAMETERS
    pretrain_length = batch_size
    memory_size = 1000000

    ### PREPROCESSING HYPERPARAMETERS
    stack_size = 4

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = False

    ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
    episode_render = True

    # Reset the graph
    tf.reset_default_graph()

    # Instantiate the DQNetwork
    dqnetwork = DQNetwork(state_size, action_size, learning_rate)

    # Instantiate memory
    memory = Memory(max_size=memory_size)

    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            state = env.reset() # TODO: Change this to work with WS env

            state, stacked_frames = stack_framesX(stacked_frames, state, False, stack_size)

        # Get the next_state, the rewards, done by tacking a random action
        choice = random.randint(1, len(possible_actions))-1
        action = possible_actions[choice]
        next_state, reward, done = env.step(onehot_to_int(action))

        if (episode_render == True):
            env.render_map()
            env.render_sub_map()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #Stack the frames
        next_state, stacked_frames = stack_framesX(stacked_frames, next_state, False, stack_size)

        # If the episode is finished (we'r dead 3x)
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experiance to memory
            memory.add((state, action, reward, next_state, done))

            # Start a new episode
            state = env.reset() # TODO: change to be able to reset the env

            # Stack the frames
            state, stacked_frames = stack_framesX(stacked_frames, state, True, stack_size)

        else:
            # Add experiance to memory
            memory.add((state, action, reward, next_state, done))

            # Our new state is new the next_state
            state = next_state

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter("/tensorboard/dqn/1")

    # Losses
    tf.summary.scalar("Loss", dqnetwork.loss)
    #tf.summary.scalar("Loss", dqnetwork.loss)

    write_op = tf.summary.merge_all()

    # Saver will help us to save out model
    saver = tf.train.Saver()

    if training == True:
        with tf.Session() as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())

            # Initialize the decay rate (that will be used to reduce epsilon)
            decay_step = 0

            for episode in range(total_episodes):
                # Set step to 0
                step = 0

                # Initialize the rewards of the episode
                episode_rewards = []
                rewards_list = []


                # Make a new episode and observe the first state
                state = env.reset() # TODO: implement reset function in Environment

                # Remember that stack frame function also calls our preprocess function.
                state, stacked_frames = stack_framesX(stacked_frames, state, True, stack_size)

                while (step < max_steps):
                    step += 1

                    # Increase decay_step
                    decay_step += 1

                    # Predict the action to take and take it
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                                 state, possible_actions, sess, dqnetwork)

                    # Perform the action and get the next_state, reward and done info
                    next_state, reward, done = env.step(onehot_to_int(action)) # TODO: some changed may needed here (changes done in environment.py)

                    if (episode_render == True):
                        env.render_map()
                        env.render_sub_map()
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Add the reward to the total reward
                    episode_rewards.append(reward)

                    # If the game is finished
                    if done:
                        # The episode ends so no next state
                        next_state = np.zeros((15, 15), dtype=np.int)

                        next_state, stack_frames = stack_framesX(stacked_frames, next_state, False, stack_size)

                        # Set step = max_steps to end the episode
                        step = max_steps

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        print('Episode: {}'.format(episode),
                              'total Reward: {}'.format(total_reward),
                              'Explore P: {:.4f}'.format(explore_probability),
                              'Training Loss {:.4f}'.format(loss))

                        rewards_list.append((episode, total_reward))

                        # Store transition <st, at, rt+1, st+1> in memory D
                        memory.add((state, action, reward, next_state, done))

                    else:
                        # Stack the frame of the next_state
                        next_state, stacked_frames = stack_framesX(stacked_frames, next_state, False, stack_size)

                        # Add experiance to memory
                        memory.add((state, action, reward, next_state, done))

                        # st+1 is now out current state
                        state = next_state

                    ### LEARNING PART
                    # Obtain random mini-batch from memory
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    # get Q values for next_state
                    Qs_next_state = sess.run(dqnetwork.output, feed_dict={dqnetwork.inputs_: next_states_mb})

                    # Set Q_ target = r is the episode ends at s+1, otherwise set Q_ target = r + gamma*maxQ(s',a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])

                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([dqnetwork.loss, dqnetwork.optimizer],
                                       feed_dict={dqnetwork.inputs_: states_mb,
                                                  dqnetwork.target_Q: targets_mb,
                                                  dqnetwork.actions_: actions_mb})

                    # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={dqnetwork.inputs_: states_mb,
                                                            dqnetwork.target_Q: targets_mb,
                                                            dqnetwork.actions_: actions_mb})

                    writer.add_summary(summary, episode)
                    writer.flush()
                    if step % 1000 == 0:
                        print("Actions taken: {}" .format(step))

                # Save model every 1 episodes
                if episode % 1 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved: {}" .format(step), "{}" .format(episode))

    print("Inference Starting")
    with tf.Session() as sess:
        total_test_rewards = []

        # Lead the model
        saver.restore(sess, "./models/model.ckpt")

        for episode in range(1):
            env.reset()
            state, reward, done = env.step(3)
            state, stacked_frames = stack_framesX(stacked_frames, state, True, stack_size)

            print("****************************************************")
            print("EPISODE ", episode)

            while True:
                # Reshape the state
                state = state.reshape((1, *state_size))
                # Get action from Q-network
                # Estimate the Qs values state
                Qs = sess.run(dqnetwork.output, feed_dict={dqnetwork.inputs_: state})

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[choice]

                print(onehot_to_int(action))

                # Perform the action and get the next_state, reward, and done information
                next_state, reward, done = env.step(onehot_to_int(action))

                if (episode_render == True):
                    env.render_map()
                    env.render_sub_map()
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                #total_rewards += reward

                '''
                if done:
                    print("Score", total_rewards)
                    total_test_rewards.append(total_rewards)
                    break
                '''

                next_state, stacked_frames = stack_framesX(stacked_frames, next_state, False, stack_size)
                state = next_state



if __name__ == '__main__':
    main()