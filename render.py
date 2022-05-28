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
glove = glove[:,3700:4000]

# qp (d4rl controls) - data (ninaweb joint numbers) pairs
pairs = {
    7: 5,
    8: 6,
    9: 7
}

for t in range(b-a):
    
    qp = np.array([0.0] * 30)
    
    #qp[2] = 1.2 # azimuth
    qp[3] = 3.14 # faceup

    for key in pairs.keys():
        value = pairs[key]-1 # matlab-to-python array conversion
        qp[key] = radians(glove[value,t]-glove[value,0])
    
    _ = env.step(qp)
    env.mj_render()