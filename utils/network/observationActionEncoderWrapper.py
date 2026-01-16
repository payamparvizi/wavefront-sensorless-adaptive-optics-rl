import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ObservationActionEncoderWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObservationActionEncoderWrapper, self).__init__(env)
        # lo = np.concatenate((env.observation_space.low, env.action_space.low))
        # ho = np.concatenate((env.observation_space.high, env.action_space.high))       
        # self.observation_space = spaces.Box(low=lo, high=ho, dtype=env.action_space.dtype)

        #observation - action space tuple
        act_space = spaces.Box(low=env.action_space.low, high=env.action_space.high, dtype=env.action_space.dtype)
        self.observation_space = gym.spaces.Dict({'obs':env.observation_space, 'action_space':act_space})

    def step(self, action):
        observation, reward, done, trunc, info = self.env.step(action)
        
        # Modify the observation using the action, for example, add the action to the observation
        modified_observation = {
            'obs': observation,
            'action_space': action
            }

        return modified_observation, reward, done, trunc, info
    
    def reset(self, seed=None, options={}):
        observation, info = self.env.reset()
        action = np.zeros(self.env.action_space.shape)

        # Modify the observation using the action, for example, add the action to the observation
        modified_observation = {
            'obs': observation,
            'action_space': action
            }
        
        return modified_observation, info

# Create an environment and wrap it with the custom wrapper
# env = gym.make('CartPole-v1')
# env = ObservationActionWrapper(env)

# # Test the environment
# observation = env.reset()
# done = False
# total_reward = 0

# while not done:
#     action = env.action_space.sample()
#     observation, reward, done, _ = env.step(action)
#     total_reward += reward

# print("Total Reward:", total_reward)
