import torch
import torch.nn as nn
import packages.tianshou as ts
import numpy as np
from packages.LipsNet.lipsnet import K_net
from torch.func import jacrev
from torch import vmap

class CustomModel_lipsnet(nn.Module):
    def __init__(self, state_shape, action_shape, obs_hidden_size=128,act_hidden_size=128, device='cpu',
                 global_lips=0, k_init=100, k_sizes=32, k_hid_act=nn.Tanh, k_out_act=nn.Softplus,
                 loss_lambda=0.1, eps=1e-4, squash_action=1):
        super(CustomModel_lipsnet, self).__init__()
        self.obs_dim = np.prod(state_shape)
        self.action_dim = np.prod(action_shape)
        conIn = obs_hidden_size + act_hidden_size
        compConIn = int(conIn/3)
        self.output_dim = compConIn
        self.device = device
        
        self.loss_lambda = loss_lambda
        self.eps = eps
        self.squash_action = squash_action

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
        
        self.k_net = K_net(global_lips, k_init, [conIn, k_sizes, int(k_sizes/2), 1], k_hid_act, k_out_act)

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
        
        # K(x) forward
        k_out = self.k_net(x)
        
        # L2 regularization backward
        if self.training and k_out.requires_grad:
            lips_loss = self.loss_lambda * (k_out ** 2).mean()
            lips_loss.backward(retain_graph=True)
        
        # f(x) forward
        f_out = self.model(x)

        # calcute jac matrix
        if k_out.requires_grad:
            jacobi = vmap(jacrev(self.model))(x)
        else:
            with torch.no_grad():
                jacobi = vmap(jacrev(self.model))(x)
        # jacobi.dim: (x.shape[0], f_out.shape[1], x.shape[1])
        #             (batch     , f output dim  , x feature dim)      

        # calcute jac norm
        jac_norm = torch.norm(jacobi, 2, dim=(1,2)).unsqueeze(1)
        
        # multi-dimensional gradient normalization (MGN)
        action = k_out * f_out / (jac_norm + self.eps)
        
        if self.squash_action == 1:
            action = torch.tanh(action)

        return action, None
