import torch
import torch.nn as nn
import packages.tianshou as ts
import numpy as np


class CustomModel(nn.Module):
    def __init__(self, state_shape, action_shape, obs_hidden_size=128,act_hidden_size=128, device='cpu'):
        super(CustomModel, self).__init__()
        self.obs_dim = np.prod(state_shape)
        self.action_dim = np.prod(action_shape)
        conIn = obs_hidden_size + act_hidden_size
        compConIn = int(conIn/3)
        self.output_dim = compConIn
        self.device = device

        # Define input heads for observation and action
        self.obs_head = nn.Sequential(
            nn.Linear(self.obs_dim, obs_hidden_size),
            nn.ReLU(),
            nn.Linear(obs_hidden_size, obs_hidden_size),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.action_dim, act_hidden_size),
            nn.ReLU(),
            nn.Linear(act_hidden_size, act_hidden_size),
            nn.ReLU(),
        )

        # Define a combiner
        self.model = nn.Sequential(
            nn.Linear(conIn, compConIn),  # Concatenated obs and state
            nn.ReLU(),
            nn.Linear(compConIn, compConIn),
            nn.ReLU(),
            nn.Linear(compConIn, compConIn),
            nn.ReLU(),
        )

    def forward(self, obs, state=None, info={}):
        cur_obs = obs['obs']
        prev_action = obs['action_space']
        if cur_obs.shape[1]>1:
            obs_feature = self.obs_head(torch.as_tensor(cur_obs,device=self.device,dtype=torch.float).flatten(1))
            action_feature = self.action_head(torch.as_tensor(prev_action,device=self.device,dtype=torch.float).flatten(1))
        else:    
            obs_feature = self.obs_head(torch.as_tensor(cur_obs,device=self.device,dtype=torch.float))
            action_feature = self.action_head(torch.as_tensor(prev_action,device=self.device,dtype=torch.float))
        # Concatenate the features from the observation and state heads
        x = torch.cat([obs_feature, action_feature], dim=-1)
        return self.model(x), None

# def main():
# env = ObservationActionEcoderWrapper(gym.make('AO-v0'))
# max_action = env.action_space.high
# net = CustomModel(env.observation_space['env_observation'].shape, env.observation_space['action_space'].shape, hidden_size=64)
# actor = ActorProb(net, env.action_space.shape[0], device='cuda').to('cuda')

# critic = Critic(
#     CustomModel(env.observation_space['env_observation'].shape, env.observation_space['action_space'].shape, hidden_size=64),
#     device='cuda'
# ).to('cuda')

# actor_critic = ActorCritic(actor, critic)

# train_envs = DummyVectorEnv([lambda: ObservationActionEcoderWrapper(gym.make('AO-v0')) for _ in range(1)])
# test_envs = DummyVectorEnv([lambda: ObservationActionEcoderWrapper(gym.make('AO-v0')) for _ in range(1)])

# for m in actor_critic.modules():
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.orthogonal_(m.weight)
#         torch.nn.init.zeros_(m.bias)

# optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)

# def dist(*logits):
#     return Independent(Normal(*logits), 1)
    
# policy = PPOPolicy(
#     actor,
#     critic,
#     optim,
#     dist)

# train_collector = Collector(
#     policy, train_envs, VectorReplayBuffer(total_size=200, buffer_num=1)
# )

# test_collector = Collector(policy, test_envs)
# trainer = onpolicy_trainer(
#     policy,
#     train_collector,
#     test_collector,
#     100,
#     200,
#     64,
#     1,
#     16,
#     step_per_collect=300,
#     # stop_fn=stop_fn,
#     test_in_train=False
#     # episode_per_test=args.episodes_per_test
# )