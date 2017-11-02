import numpy as np
import tensorflow as tf
import math

class Counter():
	def __init__(self, name):
		self.name = name
		self.counter = tf.Variable(0, name = self.name + "_counter")
		self.one = tf.constant(1, name = self.name + "_one")
		self.IncrementCounter = tf.assign_add(self.counter, self.one)

	def increment(self, session):
		session.run(self.IncrementCounter)

	def evaluate(self, session):
		return self.counter.eval(session)

	def assign(self, session, value):
		session.run(self.counter.assign(value))


class NN():
	def __init__(self, name, state_size, layers, output_size, learning_rate=0.001, gamma=0.99, log=False, grad_lim = None):
		self.state_size = state_size
		self.output_size = output_size
		self.layers = layers
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.name = name 
		self.weights = []
		self.biases = []
		self.activations = []
		self.summaries = []
		self.grad_lim = grad_lim

		self.activations.append(tf.placeholder(tf.float32, [None, self.state_size], name=self.name+"_state_input"))

		prev_dim = self.state_size

		i = 1
		if type(layers) is int:
			layers = [layers]
		for layer in layers:
			self.weights.append(tf.Variable(tf.random_uniform([prev_dim, layer], -1./math.sqrt(prev_dim), 1./math.sqrt(prev_dim)), name = self.name+"_W"+str(i)))
			self.biases.append(tf.Variable(tf.random_uniform([layer],            -1./math.sqrt(prev_dim), 1./math.sqrt(prev_dim)),name = self.name+"_b"+str(i)))
			self.activations.append(tf.nn.relu(tf.matmul(self.activations[-1], self.weights[-1]) + self.biases[-1], name=self.name+"_z"+str(i)))
			if log == True:
				self.summaries.append(tf.histogram_summary(self.name+"/W"+str(i),self.weights[-1]))
				self.summaries.append(tf.histogram_summary(self.name+"/b"+str(i),self.biases[-1]))
			prev_dim = layer
			i = i+1

		self.weights.append(tf.Variable(tf.random_uniform([prev_dim, output_size], -1./math.sqrt(prev_dim), 1./math.sqrt(prev_dim)), name = self.name+"_W"+str(i)))
		self.biases.append(tf.Variable(tf.random_uniform([output_size],            -1./math.sqrt(prev_dim), 1./math.sqrt(prev_dim)),name = self.name+"_b"+str(i)))
		self.activations.append(tf.add(tf.matmul(self.activations[-1], self.weights[-1]), self.biases[-1], name=self.name+"_out"))
		if log == True:
			self.summaries.append(tf.histogram_summary(self.name+"/W"+str(i), self.weights[-1]))
			self.summaries.append(tf.histogram_summary(self.name+"/b"+str(i), self.biases[-1]))
	

	def get_weights(self):
		return self.weights, self.biases

	def assign(self, session, Ws,bs):
		to_assign = []
		for i in xrange(len(self.weights)):
			to_assign.append(self.weights[i].assign(Ws[i]))
			to_assign.append(self.biases[i].assign(bs[i]))
		session.run(to_assign)


class DeterministicMLP(NN):
	def __init__(self, name, state_size, layers, output_size, learning_rate=0.001, gamma=0.99, log=False, L2_weight = 0, L_eps_weight = 0, grad_lim = None):
		NN.__init__(self, name, state_size, layers, output_size, learning_rate, gamma, log, grad_lim)

		self.L2_weight = L2_weight
		self.L_eps_weight = L_eps_weight

		self.y_estimate = tf.placeholder(tf.float32, [None, self.output_size], name = self.name+"_y_estimate")

		self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_estimate - self.activations[-1]),1))
		self.weight_decay = tf.add_n([tf.nn.l2_loss(var) for var in self.weights+self.biases])
		self.mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y_estimate - self.activations[-1]),1))
		self.cost = self.mse + self.L2_weight*self.weight_decay + self.L_eps_weight*self.mae
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, self.gamma,0.0,1e-6)
		self.output_optimizer = tf.train.RMSPropOptimizer(self.learning_rate, self.gamma, 0.0, 1e-6)
		if grad_lim == None:
			self.optimize = self.optimizer.minimize(self.cost)
			self.optimize_output = self.output_optimizer.minimize(self.cost, var_list=[self.weights[-1],self.biases[-1]])
		else:	
			self.gvs = self.optimizer.compute_gradients(self.cost, self.weights+self.biases)
			self.capped_gvs = [(tf.clip_by_norm(grad, self.grad_lim), var) for grad, var in self.gvs]
			self.optimize = self.optimizer.apply_gradients(self.capped_gvs)
			
			self.gvs = tf.self.output_optimizer.compute_gradients(self.cost, [self.weights[-1],self.biases[-1]])
			self.capped_gvs = [(tf.clip_by_norm(grad, self.grad_lim), var) for grad, var in self.gvs]
			self.optimize_output = self.output_optimizer.apply_gradients(self.capped_gvs)

	def evaluate(self, session, state):
		return session.run(self.activations[-1], {self.activations[0]:state})

	def train(self, session, states, labels):
		error = session.run(self.cost, {self.activations[0]:states, self.y_estimate:labels})
		session.run(self.optimize, {self.activations[0]:states, self.y_estimate:labels})
		return error

	def train_output(self, session, states, labels):
		error = session.run(self.cost, {self.activations[0]:states, self.y_estimate:labels})
		session.run(self.optimize_output, {self.activations[0]:states, self.y_estimate:labels})
		return error

	def error(self, session, states, labels):
		return session.run(self.cost, {self.activations[0]:states, self.y_estimate:labels})


class ActorMLP(NN):
	def __init__(self, name, state_size, layers, action_size, learning_rate=0.001, gamma=0.99, tau=0.1, log=False, grad_lim = None):
		NN.__init__(self, name, state_size, layers, action_size, learning_rate, gamma, log, grad_lim)
		self.tau = tau
		self.action_size = action_size

		self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
		self.net = self.weights+self.biases
		self.params_grad = tf.gradients(self.activations[-1], self.net, -self.action_gradient)
		grads = zip(self.params_grad, self.net)
		self.optimize = tf.train.RMSPropOptimizer(self.learning_rate, self.gamma, 0.0, 1e-6).apply_gradients(grads)

		self.ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)

	def target_update(self, session, Ws, bs):
		session.run(self.ema.apply(Ws+bs))

	def train(self, session, states, action_grad):
		session.run(self.optimize, {self.activations[0]: states, self.action_gradient:action_grad})

	def evaluate(self, session, states):
		return session.run(self.activations[-1], {self.activations[0]:states})

	def assign(self, session, Ws,bs):
		to_assign = []
		for i in xrange(len(self.weights)):
			to_assign.append(self.weights[i].assign(self.tau * self.weights[i] + (1-self.tau) * Ws[i]))
			to_assign.append(self.biases[i].assign(self.tau * self.biases[i] + (1-self.tau) * bs[i]))
		session.run(to_assign)

	def init_weights(self, session, Ws, bs):
		to_assign = []
		for i in xrange(len(self.weights)):
			to_assign.append(self.weights[i].assign(Ws[i]))
			to_assign.append(self.biases[i].assign(bs[i]))
		session.run(to_assign)


class CriticMLP():
	def __init__(self, name, state_size, action_size, layers, output_size, learning_rate=0.001, gamma=0.99, tau=0.1, log=False, grad_lim=None):
		self.state_size = state_size
		self.action_size = action_size
		self.output_size = output_size
		self.layers = layers
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.tau = tau
		self.name = name 
		self.weights = []
		self.biases = []
		self.activations = []
		self.summaries = []
		self.grad_lim = grad_lim

		self.activations.append(tf.placeholder(tf.float32, [None, self.state_size], name=self.name+"_state_input"))
		self.activations.append(tf.placeholder(tf.float32, [None, self.action_size], name=self.name+"_action_input"))

		prev_dim = self.state_size
	
		self.weights.append(tf.Variable(tf.random_uniform([self.state_size, layers[0]], -1./math.sqrt(self.state_size), 1./math.sqrt(self.state_size)), name = self.name+"_W1"))
		self.biases.append(tf.Variable(tf.random_uniform([layers[0]],            -1./math.sqrt(self.state_size), 1./math.sqrt(self.state_size)),name = self.name+"_b1"))
		self.activations.append(tf.nn.relu(tf.matmul(self.activations[0], self.weights[-1]) + self.biases[-1], name=self.name+"_z1"))
		if log == True:
			self.summaries.append(tf.histogram_summary(self.name+"/W1",self.weights[-1]))
			self.summaries.append(tf.histogram_summary(self.name+"/b1",self.biases[-1]))
		
		W2 = tf.Variable(tf.random_uniform([layers[0], layers[1]], -1./math.sqrt(layers[0]), 1./math.sqrt(layers[0])), name = self.name+"_W2")
		W2_action = tf.Variable(tf.random_uniform([self.action_size, layers[1]], -1./math.sqrt(self.action_size), 1./math.sqrt(self.action_size)), name = self.name+"_W2_action")
		self.weights.append(W2)
		self.weights.append(W2_action)
		self.biases.append(tf.Variable(tf.random_uniform([layers[1]],            -1./math.sqrt(layers[0]), 1./math.sqrt(layers[0])),name = self.name+"_b2"))
		self.activations.append(tf.nn.relu(tf.matmul(self.activations[-1], W2) + tf.matmul(self.activations[1], W2_action) + self.biases[-1], name=self.name+"_z2"))
		if log == True:
			self.summaries.append(tf.histogram_summary(self.name+"/W2", W2))
			self.summaries.append(tf.histogram_summary(self.name+"/W2", W2_action))
			self.summaries.append(tf.histogram_summary(self.name+"/b2", self.biases[-1]))

		self.weights.append(tf.Variable(tf.random_uniform([layers[1], output_size], -1./math.sqrt(layers[1]), 1./math.sqrt(layers[1])), name = self.name+"_W3"))
		self.biases.append(tf.Variable(tf.random_uniform([output_size],            -1./math.sqrt(layers[1]), 1./math.sqrt(layers[1])),name = self.name+"_b3"))
		self.activations.append(tf.add(tf.matmul(self.activations[-1], self.weights[-1]), self.biases[-1], name=self.name+"_out"))
		if log == True:
			self.summaries.append(tf.histogram_summary(self.name+"/W3", self.weights[-1]))
			self.summaries.append(tf.histogram_summary(self.name+"/b3", self.biases[-1]))
				
		
		self.y_estimate = tf.placeholder(tf.float32, [None, 1], name=self.name+"_y_estimate")

		self.cost = tf.reduce_mean(tf.square(self.y_estimate - self.activations[-1]))
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, self.gamma, 0.0, 1e-6)
		if grad_lim == None:
			self.optimize = self.optimizer.minimize(self.cost)
		else:	
			self.gvs = self.optimizer.compute_gradients(self.cost, self.weights+self.biases)
			self.capped_gvs = [(tf.clip_by_norm(grad, self.grad_lim), var) for grad, var in self.gvs]
			self.optimize = self.optimizer.apply_gradients(self.capped_gvs)

		self.action_grads = tf.gradients(self.activations[-1], self.activations[1])

		self.ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)
		


	def get_weights(self):
		return self.weights, self.biases

	def gradients(self, session, states, actions):
		return session.run(self.action_grads, {self.activations[0]:states, self.activations[1]:actions})[0]

	def train(self, session, states, actions, labels):
		error = session.run(self.cost, {self.activations[0]:states, self.activations[1]:actions, self.y_estimate:labels})
		session.run(self.optimize, {self.activations[0]:states, self.activations[1]:actions, self.y_estimate:labels})
		return error
	
	def evaluate(self, session, states, actions):
		return session.run(self.activations[-1], {self.activations[0]:states, self.activations[1]:actions})

	def target_update(self,session, Ws,bs):
		session.run(self.ema.apply(Ws+bs))
	
	def assign(self, session, Ws,bs):
		to_assign = []
		for i in xrange(len(self.biases)):
			to_assign.append(self.biases[i].assign(self.tau * self.biases[i] + (1-self.tau) * bs[i]))
			to_assign.append(self.weights[i].assign(self.tau * self.weights[i] + (1-self.tau) * Ws[i]))
		to_assign.append(self.weights[i+1].assign(self.tau * self.weights[i+1] + (1-self.tau) * Ws[i+1]))
		session.run(to_assign)

	def init_weights(self, session, Ws, bs):
		to_assign = []
		for i in xrange(len(self.weights)):
			to_assign.append(self.weights[i].assign(Ws[i]))
		for i in xrange(len(self.biases)):
			to_assign.append(self.biases[i].assign(bs[i]))
		session.run(to_assign)


class CriticNetwork1(object):
	def __init__(self, sess, state_size, action_size, layers, BATCH_SIZE, TAU, LEARNING_RATE, L2):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.L2 = L2
		self.layers = layers

	        # CRITIC
        	self.state, self.action, self.out, self.net = \
		self.create_critic_network(state_size, action_size)

	        # TARGET CRITIC
        	self.target_state, self.target_action, self.target_update, self.target_net, self.target_out = self.crate_critic_target_network(
		state_size, action_size, self.net)

		# TRAINING
        	self.y = tf.placeholder("float", [None, 1])
	        self.error = tf.reduce_mean(tf.square(self.y - self.out))
	        self.weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.net])
	        self.loss = self.error + self.weight_decay
	        self.optimize = tf.train.AdamOptimizer(
	        LEARNING_RATE).minimize(self.loss)

        	# GRADIENTS for policy update
	        self.action_grads = tf.gradients(self.out, self.action)

        	# INIT VARIABLES
	        self.sess.run(tf.initialize_all_variables())

	def gradients(self, states, actions):
        	return self.sess.run(self.action_grads, feed_dict={self.state: states, self.action: actions})[0]

	def train(self, y, states, actions):
        	self.sess.run(self.optimize, feed_dict={self.y: y, self.state: states, self.action: actions})

	def predict(self, states, actions):
		return self.sess.run(self.out, feed_dict={self.state: states, self.action: actions})

	def target_predict(self, states, actions):
        	return self.sess.run(self.target_out, feed_dict={self.target_state: states, self.target_action: actions})

	def target_train(self):
        	self.sess.run(self.target_update)

	def crate_critic_target_network(self, input_dim, action_dim, net):
		# input
		state = tf.placeholder(tf.float32, shape=[None, input_dim])
		action = tf.placeholder(tf.float32, shape=[None, action_dim])

		ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]
	
		h1 = tf.nn.relu(tf.matmul(state, target_net[0]) + target_net[1])
		h2 = tf.nn.relu(tf.matmul(h1, target_net[2]) + tf.matmul(action, target_net[3]) + target_net[4])
		out = tf.identity(tf.matmul(h2, target_net[5]) + target_net[6])
		
		return state, action, target_update, target_net, out
	
	def create_critic_network(self, state_dim, action_dim):
	        # input
	        state = tf.placeholder(tf.float32, shape=[None, state_dim])
	        action = tf.placeholder(tf.float32, shape=[None, action_dim])

		# network weights
		W1 = self.weight_variable([state_dim, self.layers[0]])
		b1 = self.bias_variable([self.layers[0]])
		W2 = self.weight_variable([self.layers[0], self.layers[1]])
		b2 = self.bias_variable([self.layers[1]])
		W2_action = self.weight_variable([action_dim, self.layers[1]])
		W3 = self.weight_variable([self.layers[1], 1])
		b3 = self.bias_variable([1])
	
		# computation
		h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
		h2 = tf.nn.relu(tf.matmul(h1, W2) + tf.matmul(action, W2_action) + b2)
		out = tf.identity(tf.matmul(h2, W3) + b3)

        	return state, action, out, [W1, b1, W2, W2_action, b2, W3, b3]

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.001)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.001, shape=shape)
		return tf.Variable(initial)

class ActorNetwork1(object):
	def __init__(self, sess, state_size, action_size, layers, BATCH_SIZE, TAU, LEARNING_RATE, L2):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.L2 = L2
		self.layers = layers

		# ACTOR
		self.state, self.out, self.net = self.create_actor_network(state_size, action_size)

		# TARGET NETWORK
		self.target_state, self.target_update, self.target_net, self.target_out = self.crate_actor_target_network(state_size, self.net)


		# TRAINING
		self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
		self.params_grad = tf.gradients(self.out, self.net, -self.action_gradient)
		grads = zip(self.params_grad, self.net)
		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

		# INIT VARIABLES
		self.sess.run(tf.initialize_all_variables())


	def train(self, states, action_grads):
		self.sess.run(self.optimize, feed_dict={self.state: states,self.action_gradient: action_grads})

	def predict(self, states):
		return self.sess.run(self.out, feed_dict={self.state: states})

	def target_predict(self, states):
		return self.sess.run(self.target_out, feed_dict={self.target_state: states})

	def target_train(self):
		self.sess.run(self.target_update)

	def crate_actor_target_network(self, input_dim, net):
		# input
		state = tf.placeholder(tf.float32, shape=[None, input_dim])

		ema = tf.train.ExponentialMovingAverage(decay=1-self.TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		h1 = tf.nn.relu(tf.matmul(state, target_net[0]) + target_net[1])
		h2 = tf.nn.relu(tf.matmul(h1, target_net[2]) + target_net[3])
		out = tf.identity(tf.matmul(h2, target_net[4]) + target_net[5])

		return state, target_update, target_net, out

	def create_actor_network(self, input_dim, output_dim):
		# input
		state = tf.placeholder(tf.float32, shape=[None, input_dim])

		# network weights
		W1 = self.weight_variable([input_dim, self.layers[0]])
		b1 = self.bias_variable([self.layers[0]])
		W2 = self.weight_variable([self.layers[0], self.layers[1]])
		b2 = self.bias_variable([self.layers[1]])
		W3 = self.weight_variable([self.layers[1], output_dim])
		b3 = self.bias_variable([output_dim])

		# computation
		h1 = tf.nn.relu(tf.matmul(state,W1) + b1)
		h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)
		out = tf.identity(tf.matmul(h2,W3) + b3)

		return state, out, [W1, b1, W2, b2, W3, b3]

	def weight_variable(self, shape):
		initial = tf.random_uniform(shape, minval=-0.05, maxval=0.05)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape=shape)
		return tf.Variable(initial)

