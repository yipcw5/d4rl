import gym
import d4rl
from mujoco_env import MujocoEnv

import numpy as np
from math import radians

env = gym.make('door-v0')
env.mujoco_render_frames = True
obs = env.reset()

for t in range(500):
    qp = np.array([0.0] * 30)
    #qp[2] = 1.57
    qp[1] = -1.57 # vertical
    qp[3] = 0.7 # faceup
    if t < 300: s = t
    else: s = 300
    qp[7] += radians(136-117) * s/300
    qp[8] += radians(171-95) * s/300
    qp[9] += radians(140-65) * s/300
    _ = env.step(qp)
    #print('www',env.action_space.sample())
    env.mj_render()
    #print('qp', qp)
    #env.step(1.0)
    #env.mj_render()

#env.action_space.sample()