import numpy as np
import os

class ReplayBuffer():

	populated = 0

	def __init__(self, state_space, action_space, buffer_size):
		self.buffer_size = buffer_size+1
		self.state_size = state_space
		self.action_size = action_space

		self.EmptyBuffer()

	def SampleMiniBatch(self, batch):
		top = min(batch, self.populated)
		batch = np.arange(self.populated)
		np.random.shuffle(batch)
		batch = batch[:top]
		R = self.R[batch]
		S = self.S[batch]
		A = self.A[batch]
		D = self.D[batch]
		Stag = self.Stag[batch]
		return S, A, R, Stag, D


	def StoreTransition(self, s_t, a_t, r_t, s_t_next, d_t=0):
		s_t = s_t.reshape(1, self.state_size)
		s_t_next = s_t_next.reshape(1, self.state_size)
		a_t = a_t.reshape(1, self.action_size)
		r_t = r_t.reshape(1, 1)
		d_t = np.array([d_t]).reshape(1, 1)
		
		self.S = np.concatenate((self.S, s_t))
		self.Stag = np.concatenate((self.Stag, s_t_next))
		self.A = np.concatenate((self.A, a_t))
		self.R = np.concatenate((self.R, r_t))
		self.D = np.concatenate((self.D, d_t))

		if self.populated < self.buffer_size:
			if self.populated == 0:
				self.S = np.delete(self.S,0,0)
				self.A = np.delete(self.A,0,0)
				self.R = np.delete(self.R,0,0)
				self.Stag = np.delete(self.Stag,0,0)
				self.D = np.delete(self.D,0,0)
			self.populated += 1
		else:
			self.S = np.delete(self.S,0,0)
			self.A = np.delete(self.A,0,0)
			self.R = np.delete(self.R,0,0)
			self.Stag = np.delete(self.Stag,0,0)
			self.D = np.delete(self.D,0,0)

	def SaveBuffer(self, save_file):
		np.savez_compressed(save_file, S=self.S, A=self.A, R=self.R, Stag=self.Stag, D=self.D, \
		pop=self.populated, bs=self.buffer_size, ss=self.state_size, acs=self.action_size)

	def LoadBuffer(self, save_file):
		if not os.path.exists(save_file+'.npz'):
			return False
		arrays = np.load(save_file+'.npz')
		self.S = arrays['S']
		self.A = arrays['A']
		self.R = arrays['R']
		self.Stag = arrays['Stag']
		self.D = arrays['D']
		self.populated = arrays['pop'].reshape(1)[0]
		self.buffer_size = arrays['bs'].reshape(1)[0]
		self.state_space = arrays['ss'].reshape(1)[0]
		self.action_space = arrays['acs'].reshape(1)[0]
		return True
		
	def EmptyBuffer(self):
		self.R = np.zeros([1,1])
		self.A = np.zeros([1,self.action_size])
		self.S = np.zeros([1,self.state_size])
		self.Stag = np.zeros([1,self.state_size])
		self.D = np.zeros([1,1])
		self.populated = 0
	
	def GetOccupency(self):
		return self.populated





