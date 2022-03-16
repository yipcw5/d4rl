import gym
import d4rl 


env = gym.make('door-v0')
env.mujoco_render_frames = True
obs = env.reset()

for t in range(500):
    _ = env.step(env.action_space.sample())
    env.mj_render()

#env.action_space.sample()