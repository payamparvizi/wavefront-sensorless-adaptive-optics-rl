import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ObservationEncoderStackingWrapper(gym.Wrapper):
    def __init__(self, env, stack_num):
        super(ObservationEncoderStackingWrapper, self).__init__(env)
        self.stack_num = stack_num
        self.stack_obs = None

        # Update observation space to include stacking
        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low,(self.stack_num,1)),
            high=np.tile(env.observation_space.high,(self.stack_num,1)),
            shape=(self.stack_num, *env.observation_space.shape),
            dtype=env.observation_space.dtype
        )

    def reset(self, seed=None, options={}):
        observation, info = self.env.reset()
        self.stack_obs = np.zeros_like(self.observation_space.low, dtype=observation.dtype)
        self.stack_obs[-1] = observation

        return self.stack_obs, info

    def step(self, action):
        observation, reward, done, trunc, info = self.env.step(action)
        self.stack_obs = np.roll(self.stack_obs, shift=-1, axis=0)
        self.stack_obs[-1] = observation

        return self.stack_obs, reward, done, trunc, info

