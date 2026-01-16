import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ObservationActionEncoderStackingWrapper(gym.Wrapper):
    def __init__(self, env, stack_num):
        super(ObservationActionEncoderStackingWrapper, self).__init__(env)
        self.stack_num = stack_num
        self.stack_obs = None
        self.stack_act = None
        # print(np.tile(env.unwrapped.observation_space.low,(self.stack_num,1)))
        # print(np.tile(env.unwrapped.observation_space.high,(self.stack_num,1)))
        # Update observation space to include stacking
        env_obs_space = gym.spaces.Box(
            # low=np.tile(env.unwrapped.observation_space.low,(self.stack_num,1)),
            # high=np.tile(env.unwrapped.observation_space.high,(self.stack_num,1)),
            low=np.tile(env.observation_space['obs'].low,(self.stack_num,1)),
            high=np.tile(env.observation_space['obs'].high,(self.stack_num,1)),
            shape=(self.stack_num, *env.observation_space['obs'].shape),
            dtype=env.observation_space['obs'].dtype
        )
        env_action_space = gym.spaces.Box(
            # low=np.tile(env.unwrapped.observation_space.low,(self.stack_num,1)),
            # high=np.tile(env.unwrapped.observation_space.high,(self.stack_num,1)),
            low=np.tile(env.observation_space['action_space'].low,(self.stack_num,1)),
            high=np.tile(env.observation_space['action_space'].high,(self.stack_num,1)),
            shape=(self.stack_num, *env.observation_space['action_space'].shape),
            dtype=env.observation_space['action_space'].dtype
        )
        self.observation_space = gym.spaces.Dict({'obs':env_obs_space, 'action_space':env_action_space})

    def reset(self, seed=None, options={}):
        observation, info = self.env.reset()
        self.stack_obs = np.zeros_like(self.observation_space['obs'].low, dtype=observation['obs'].dtype)
        self.stack_obs[-1] = observation['obs']

        self.stack_act = np.zeros_like(self.observation_space['action_space'].low, dtype=observation['action_space'].dtype)
        self.stack_act[-1] = observation['action_space']
        return {'obs':self.stack_obs, 'action_space':self.stack_act}, info

    def step(self, action):
        observation, reward, done, trunc, info = self.env.step(action)
        self.stack_obs = np.roll(self.stack_obs, shift=-1, axis=0)
        self.stack_obs[-1] = observation['obs']

        self.stack_act = np.roll(self.stack_act, shift=-1, axis=0)
        self.stack_act[-1] = observation['action_space']
        return {'obs':self.stack_obs, 'action_space':self.stack_act}, reward, done, trunc, info

