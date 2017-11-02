import sys
import numpy as np
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
import gym
import math
from collections import deque
from utils.Networks import *
from utils.OUnoise import *

# global definitions
OUT_DIR = "results5_down3_y35/"
Q_NET_SIZES = (40,100)
STATE_SIZE = 5
ACTION_SIZE = 1
DISCRETIZATION = 5
NOISE = 5
LEARNING_RATE = 0.0001
GAMMA = 0.999
ANNEALING = 1000		# (5 episodes)
EPSILON = 0.1
BUFFER_SIZE = 200000
MINI_BATCH = 32
BATCHES = 2
OBSERVATION_PHASE = 600		# (3 episdoes)
ENVIRONMENT = 'CartPole-v1'
EPISODES = 3
STEPS = 300
SAVE_RATE = 100
C_STEPS = 20000		# (30 episodes)
BUFFER_FILE = 'Replay_buffer'
DISPLAY = True

##################
# counter
##################
save_counter = tf.Variable(0, name="save_counter")
log_counter = tf.Variable(0, name = "log_counter")
one = tf.constant(1, name="one")
IncrementSaveCounter = tf.assign_add(save_counter,one)
IncrementLogCounter = tf.assign_add(log_counter,one)

##################
# episode rewards
#################
reward = tf.placeholder(tf.float32, [1], name="reward")
save_reward = reward[0]
reward_sum = tf.scalar_summary('Reward',save_reward)

hot_ind = tf.placeholder(tf.int64, [None])
hot1 = tf.one_hot(hot_ind, DISCRETIZATION, on_value=1, off_value=0)

##################
# Q networks
##################
state_input = tf.placeholder(tf.float32, [None, STATE_SIZE], name="Q_state_input")

prevDim = STATE_SIZE
prevOut = state_input

Q = DeterministicMLP("Q", STATE_SIZE, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True, 0.0, 0.0)
Q_target = DeterministicMLP("Q_target", STATE_SIZE, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True, 0.0, 0.0)

# training procedure
y_estimate = tf.placeholder(tf.float32, [None, DISCRETIZATION], name = "y_estimate")

saver = tf.train.Saver()
init = tf.initialize_all_variables()

sess = tf.Session()


# initialize variables (and target network)
sess.run(init)
Ws, bs = Q.get_weights()
Q_target.assign(sess, Ws, bs)

# initialize environment
env = gym.make(ENVIRONMENT)

# initialize replay buffer
R = ReplayBuffer(STATE_SIZE, ACTION_SIZE, BUFFER_SIZE)
buf = R.LoadBuffer(OUT_DIR+BUFFER_FILE)
if buf:
	populated = R.GetOccupency()
	print("Replay buffer loaded from disk, occupied: " + str(populated))

# load saved model
ckpt = tf.train.get_checkpoint_state(OUT_DIR)
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess,ckpt.model_checkpoint_path)
	print("Model loaded from disk")

# define action discretization
a_max = env.action_space.high
a_min = env.action_space.low
actions1 = np.linspace(a_min, a_max, DISCRETIZATION)
actions_deque = deque(actions1)

ann_fric = (1-EPSILON)/ANNEALING
EXP_PROB = 1


def find_descrete_action(d, value):
	dist = np.abs(d[0] - value)
	ind = 0
	for i in xrange(len(d)):
		if np.abs(d[i] - value) < dist:
			dist = np.abs(d[i] - value)
			ind = i
	return ind



# main learning loop
print("Starting to learn in environment: " + ENVIRONMENT)
steps = 0
for episode_i in xrange(1,EPISODES+1):
	st = env.reset()
	totalR = 0
	for t in xrange(1,STEPS+1):
		if DISPLAY:
			env.render()

		# select action
		q_vector = Q.evaluate(sess, st.reshape(1,STATE_SIZE))
		a_index = np.argmax(q_vector)
		at = actions_deque[a_index]

		# execute action
		st_next, rt, Done, _ = env.step(np.array([at]))
		if Done:
			dt = 1
		else:
			dt = 0
		totalR += rt
		st = st_next
	
	r = np.array([totalR])
	sess.run(IncrementLogCounter)
	print ("episode %d/%d (%d), reward: %f" % (episode_i, EPISODES, log_counter.eval(sess), totalR))

