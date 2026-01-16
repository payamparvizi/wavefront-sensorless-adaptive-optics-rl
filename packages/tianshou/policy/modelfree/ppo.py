from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn

from packages.tianshou.data import Batch, ReplayBuffer, to_torch_as
from packages.tianshou.policy import A2CPolicy
from packages.tianshou.utils.net.common import ActorCritic
from torch.distributions import Normal, Independent

from torch.fft import fft
import copy


class PPOPolicy(A2CPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        
        ar_case: int = 0, 
        
        lambda_T: float = 2.0,
        lambda_S: float = 1.0,
        sigma_s_bar: float = 10,
        
        lambda_P: float = 2.0,
        c_homog: float = 5,
        noise_pym: float = 1e-12,
        
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic
        
        # action regularization case
        self._ar_case = ar_case
        
        # Mysore parameters
        self._sigma_s_bar = sigma_s_bar
        self._lambda_T = lambda_T
        self._lambda_S = lambda_S
        
        # pym parameters
        self._c_homog = c_homog
        self._lambda_P = lambda_P
        self._c_homog_bool = True
        self._lambda_bool = True
        self._noise_pym = noise_pym
        
        
    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            batch.logp_old = self(batch).dist.log_prob(batch.act)
        return batch

    def action_sampling(self, obs):
        (mean, std), _ = self.actor(obs)
        P = Normal(mean, std)
        P = Independent(P, 1) 
        act = P.sample()
        log_prob = P.log_prob(act)
        return mean, act, log_prob.exp()
        
    
    def ar_caps_fun(self, batch: Batch) -> Batch:
        mean, act, prob_act = self.action_sampling(batch.obs)
        mean_next, act_next, prob_act_next = self.action_sampling(batch.obs_next)
        
        batch_obs_copy = copy.deepcopy(batch.obs)
        obs_bar = np.random.normal(loc=batch_obs_copy.obs, scale=self._sigma_s_bar)
        batch_obs_copy.obs = obs_bar
        
        mean_bar, act_bar, prob_act_bar = self.action_sampling(batch_obs_copy)
        
        # calculate the Euclidean distance for temporal and spatial smoothness:
        DT = torch.norm(mean - mean_next, p=2, dim=-1)
        DS = torch.norm(mean - mean_bar, p=2, dim=-1)
        
        J_mysore = self._lambda_T * DT + self._lambda_S * DS
        return J_mysore.mean()


    def ar_aps_fun(self, batch: Batch) -> Batch:
        mean, act, prob_act = self.action_sampling(batch.obs)
        mean_next, act_next, prob_act_next = self.action_sampling(batch.obs_next)
        
        obs = torch.tensor(batch.obs.obs)
        obs_next = torch.tensor(batch.obs_next.obs)
        
        noise_size = mean.shape[0]
        noise = self._noise_pym * torch.abs(torch.randn(noise_size))
        noise2 = self._noise_pym * torch.abs(torch.randn(noise_size))
        
        # calculate the Euclidean distance for temporal and spatial smoothness:
        DT = torch.norm(mean - mean_next, p=2, dim=-1).cpu() + noise
        DO = torch.norm(obs - obs_next, p=2, dim=-1).cpu().mean(dim=-1) + noise2
        
        DP = abs(torch.log(self._c_homog * DT/DO))
        
        J_aps = self._lambda_P * DP
        return J_aps.mean()


    def action_fluctuation(self, batch: Batch) -> Batch:
        mean, act, prob_act = self.action_sampling(batch.obs)
        mean_next, act_next, prob_act_next = self.action_sampling(batch.obs_next)
        
        obs = torch.tensor(batch.obs.obs)
        obs_next = torch.tensor(batch.obs_next.obs)
        
        act_fluc = torch.norm(act - act_next, p=2, dim=-1).cpu()
        mu_fluc = torch.norm(mean - mean_next, p=2, dim=-1).cpu()
        obs_fluc = torch.norm(obs - obs_next, p=2, dim=-1).cpu().mean(dim=-1)
        
        K_act = act_fluc/obs_fluc
        K_mu = mu_fluc/obs_fluc
        
        return act_fluc.mean(), K_act.mean(), K_mu.mean(), obs_fluc.mean()


    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses, act_flucs, K_acts, K_means, obs_flucs = [], [], [], [], [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                
                # calculate loss for actor
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv -
                                     mean) / (std + self._eps)  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) -
                         minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                    
                act_fluc, K_act, K_mean, obs_fluc = self.action_fluctuation(minibatch)
                
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                policy_loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                
                if self._ar_case == 0:   # the standard PPO algorithm
                    loss = policy_loss
                
                elif self._ar_case == 1:   # the standard PPO + Mysore
                    # calculate the action regularization:
                    J_mysore = self.ar_caps_fun(minibatch)
                    loss = policy_loss + J_mysore
                
                elif self._ar_case == 2:   # the standard PPO + pym
                    # calculate the action regularization:
                    J_aps = self.ar_aps_fun(minibatch)
                    loss = policy_loss + J_aps
                                    
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
                act_flucs.append(act_fluc.item())
                obs_flucs.append(obs_fluc.item())
                K_acts.append(K_act.item())
                K_means.append(K_mean.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            "act_fluctuation": act_flucs,
            "obs_fluctuation": obs_flucs,
            "Lipschitz_const_stochastic": K_acts,
            "Lipschitz_const_deterministic": K_means,
        }
