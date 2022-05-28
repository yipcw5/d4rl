import gym
import d4rl
from mujoco_env import MujocoEnv

import numpy as np
from math import radians
from scipy.io import loadmat

env = gym.make('door-v0')
env.mujoco_render_frames = True
obs = env.reset()

# load hand model dataset
data = loadmat('data.mat')

# get finger joint data
# data['glove'] size: 101014x22 (timestep x joint_number)
glove = np.transpose(data['glove'])

# metacarpophalangeal (joint1), proximal phalangeal (joint2) and distal phalangeal joints (joint3)
# hardcoded time period a-b of joint angles to simulate, taken from manual inspection of matlab plot
a = 3700
b = 4000

meta, prox, dist = glove[4][a:b], glove[5][a:b], glove[6][a:b]

for t in range(b-a):
    
    qp = np.array([0.0] * 30)
    
    qp[2] = 1.2 # azimuth
    #qp[1] = -1.57 # vertical
    qp[3] = 1.57 # faceup

    qp[7] = radians(meta[t]-meta[0])
    qp[8] = radians(prox[t]-prox[0])
    qp[9] = radians(dist[t]-dist[0])
    
    '''
    qp[7] += radians(136-117) * s/iters #base joint
    qp[8] += radians(171-95) * s/iters #middle joint
    qp[9] += radians(140-65) * s/iters #top joint
    '''
    _ = env.step(qp)
    #print('www',env.action_space.sample())
    env.mj_render()
    #print('qp', qp)
    #env.step(1.0)
    #env.mj_render()

#env.action_space.sample()