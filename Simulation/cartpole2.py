"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class CartPoleEnv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
	
	# yotams parameters
        self.masscart = 0.496
        self.masspole = 0.231
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.314 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.tau = 0.035  # seconds between state updates


	# original parameters
        #self.masscart = 1.0
        #self.masspole = 0.1
        #self.total_mass = (self.masspole + self.masscart)
        #self.length = 0.5 # actually half the pole's length
        #self.polemass_length = (self.masspole * self.length)
        #self.tau = 0.02  # seconds between state updates

        self.force_mag = 10.0
        
# Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 1.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([self.x_threshold, np.inf, self.theta_threshold_radians * 2, np.inf])
	self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
	reward = self._calc_reward(x, x_dot, theta, theta_dot, action[0])
	force = np.clip(action, -self.force_mag, self.force_mag)[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        return self._get_obs(), reward, False, {}

    def _reset(self):
	margin = 1
	temp = self.np_random.uniform(low=-margin, high=margin, size=(3,))
	angle = np.pi + self.np_random.uniform(low=-margin, high=margin, size=(1,))
	self.state = np.array([temp[0], temp[1], angle, temp[2]])
        #self.state = self.np_random.uniform(low=-margin, high=margin, size=(4,))
        self.steps_beyond_done = None
        return self._get_obs() 

    def _get_obs(self):
	x, x_dot, theta, theta_dot = self.state
	return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])

    def _calc_reward(self, x, x_dot, theta, theta_dot, u):
	q = [5,0.1,7.5,0.1]
	r = 0.01
	n = 0.1
	#cost = n*(q[0]*np.power(x,2) + q[1]*np.power(x_dot,2) + q[2]*np.power(1-np.cos(theta),2) + q[3]*np.power(theta_dot,2) + r*u**2)
	cost = (np.abs(1-np.cos(theta))) + (x>5) + (x<-5)

	#cost = n*(q[0]*np.abs(x) + q[1]*np.abs(x_dot) - q[2]*(2-np.abs(1-np.cos(theta))) + q[3]*np.abs(theta_dot) + r*np.abs(u))    #THIS WAS THE FINAL COST
	#reward = -n*(q[0]*np.power(x,2) + q[1]*np.power(x_dot,2) + q[1]*np.power(np.cos(theta),2) + q[2]*np.power(np.sin(theta),2) + q[3]*np.power(theta_dot,2))
	return -cost

    def _angle_normalize(self, x):
	return (((x+np.pi) % (2*np.pi)) - np.pi)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 1000
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
