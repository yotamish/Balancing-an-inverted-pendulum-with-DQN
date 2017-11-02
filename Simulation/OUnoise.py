import numpy as np

class OUnoise():
	theta = 0.15
	sigma = 0.3
	mu = 0
	x = 0
	W_mu = 0
	W_sigma = 0.3
	dt = 0.05

	def __init__(self, action_dim, theta=0.15, sigma=0.3, mu=0, dt=0.05):
		self.theta = theta
		self.sigma = sigma
		self.mu = mu
		self.action_dim = action_dim
		self.state = np.ones(self.action_dim) * self.mu
		self.dt = dt

	def Reset(self):
		self.state = np.ones(self.action_dim) * self.mu

	def Sample(self):
		self._advance()
		return self._noise()

	def _advance(self):
		W = np.random.normal(self.W_mu, self.W_sigma)
		dt = self.dt
		d_state = self.theta * (self.state - self.mu) * dt + self.sigma * W * dt
		self.state = self.state + d_state 

	def _noise(self):
		return self.state




