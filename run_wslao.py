#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import pickle

import numpy as np
import torch
import gymnasium as gym
import gym_AO

from utils.network.observationActionEncoderStackingWrapper import ObservationActionEncoderStackingWrapper
from utils.network.observationActionEncoderWrapper import ObservationActionEncoderWrapper
from utils.network.observationEncoderStackingWrapper import ObservationEncoderStackingWrapper
from utils.network.customAgent import CustomModel
from utils.network.customAgent_obs import CustomModel_obs
from utils.network.customAgentLipsNet import CustomModel_lipsnet

from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from packages.tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from packages.tianshou.policy import PPOPolicy
from packages.tianshou.policy.base import BasePolicy
from packages.tianshou.trainer import OnpolicyTrainer
from packages.tianshou.utils.net.common import ActorCritic, Net
from packages.tianshou.utils.net.continuous import ActorProb, Critic
from packages.tianshou.env import DummyVectorEnv
from packages.tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from utils.arguments.arguments import get_args

def env_config(args, layer_id):

    env_ = gym.make(args.task,
                    atm_type = args.atm_type,                                           # atmospheric condition: 'quasi_static', 'semi_dynamic', 'dynamic'
                    atm_vel = args.atm_vel,                                             # atmosphere velocity
                    atm_fried = args.atm_fried,                                         # Fried parameter of the atmosphere
                    act_type = args.act_type,                                           # action type: 'num_actuators', 'zernike'
                    act_dim = args.act_dim,                                             # action dimension
                    obs_dim = args.obs_dim,                                             # observation dimension
                    rew_type = args.rew_type,                                           # reward type: 'strehl_ratio', 'smf_ssim'
                    rew_threshold = args.threshold,                                     # Threshould of the reward value
                    timesteps_per_episode = 20,                                         # Number of timesteps per episode
                    flat_mirror_start_per_episode = args.flat_mirror_start_per_episode, # If we want each episode to start with flat mirror
                    SH_operation = args.SH_operation,                                   # If we require Shack_Hartmann wavefront sensor operation
                    delta_t = args.delta_t,
                    c_act_range = args.c_act_range,
                    c_rand = args.c_rand,
                    c_mult = args.c_mult,
                    c_rew = args.c_rew,
                    c_mode1 = args.c_mode1,
                    seed_v = args.seed,
                    layer_no = layer_id
                    )
    return env_


def test_ppo(args: argparse.Namespace = get_args()) -> None:
    
    args.test_num = args.training_num
    
    env = ObservationActionEncoderStackingWrapper(ObservationActionEncoderWrapper(env_config(get_args(), 0)),args.stack_num)
    train_envs = DummyVectorEnv([lambda i=i: ObservationActionEncoderStackingWrapper(ObservationActionEncoderWrapper(env_config(get_args(), i+1)),args.stack_num) for i in range(args.training_num)])
    test_envs = train_envs
    
    args.env_observation = env.observation_space['obs'].shape 
    args.action_space = env.observation_space['action_space'].shape

    
    args.action_shape = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]
    
    
    args.act_hidden_size = args.hidden_size
    args.obs_hidden_size = args.hidden_size
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    if args.regularization_case == "standard_PPO":
        args.ar_case = 0
        ann_type = "mlp_network"
    elif args.regularization_case == "PPO_CAPS":
        args.ar_case = 1
        ann_type = "mlp_network"
    elif args.regularization_case == "PPO_APS":
        args.ar_case = 2
        ann_type = "mlp_network"
    elif args.regularization_case == "PPO_LipsNet":
        args.ar_case = 0
        ann_type = "lipsnet_network"
        
    # model
    if ann_type == "mlp_network":
        net = CustomModel(args.env_observation, args.action_space, 
                          obs_hidden_size=args.obs_hidden_size, 
                          act_hidden_size=args.act_hidden_size, 
                          device=args.device)
        
    elif ann_type == "lipsnet_network":
        net = CustomModel_lipsnet(args.env_observation, args.action_space, 
                          obs_hidden_size=args.obs_hidden_size, 
                          act_hidden_size=args.act_hidden_size, 
                          device=args.device, 
                          global_lips=args.global_lips, k_init=args.k_init, k_sizes=args.k_sizes, 
                          k_hid_act=nn.Tanh, k_out_act=nn.Softplus,
                          loss_lambda=args.loss_lambda, eps=args.eps_k_net, squash_action=args.squash_action)
    
    
    net_critic = CustomModel(args.env_observation, args.action_space, 
                             obs_hidden_size=args.obs_hidden_size, 
                             act_hidden_size=args.act_hidden_size, 
                             device=args.device)
        
    actor = ActorProb(net, args.action_shape, max_action=args.max_action, device=args.device).to(args.device)
    critic = Critic(net_critic, device=args.device).to(args.device)
    
    actor_critic = ActorCritic(actor, critic)
    
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    # K-learning rate is added in LipsNet network:
    if ann_type == "mlp_network":
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    
    elif ann_type == "lipsnet_network":
        k_net_params = list(actor_critic.actor.preprocess.k_net.parameters())
        k_net_param_ids = set(id(p) for p in k_net_params)
        
        other_params = [p for p in actor_critic.parameters() if id(p) not in k_net_param_ids]
        
        assert len(set(actor_critic.parameters())) == len(k_net_params) + len(other_params), "Parameter split mismatch"
        assert set(id(p) for p in k_net_params).isdisjoint(id(p) for p in other_params), "Parameter overlap"
        
        optim = torch.optim.Adam([
            {'params': other_params, 'lr': args.lr},
            {'params': k_net_params, 'lr': args.lr_k_net}
            ])


    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)


    def dist(*logits):
        return Independent(Normal(*logits), 1)
        
    policy: PPOPolicy = PPOPolicy(
        actor=actor, 
        critic=critic, 
        optim=optim, 
        dist_fn=dist, 
        discount_factor=args.gamma, 
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip, 
        vf_coef=args.vf_coef, 
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv, 
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip, 
        gae_lambda=args.gae_lambda,  
        action_space=env.action_space, 
        deterministic_eval=args.deterministic_eval,
        action_scaling=False, 
        ar_case=args.ar_case,
        lambda_T=args.lambda_T,
        lambda_S=args.lambda_S,
        sigma_s_bar=args.sigma_s_bar,
        c_homog = args.c_homog,
        lambda_P = args.lambda_P,
        noise_pym = args.noise_pym
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    if args.saved_policy == 1:
        log_name_policy = os.path.join('policies', 'quasi_static')
        log_path_policy = os.path.join('logs_saved', log_name_policy)
        ckpt_path = os.path.join(log_path_policy, "policy_2.torch")
        state_dict = torch.load(ckpt_path, map_location=args.device)
        policy.load_state_dict(state_dict)
        
        
    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
        
    train_collector = Collector(policy=policy, env=train_envs, buffer=buffer, 
                                exploration_noise=True)
    
    test_collector = Collector(policy=policy, env=test_envs)
    
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.ar_case), str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    logger = WandbLogger(
    save_interval= 1,
    train_interval = 2,
    test_interval = 1,
    update_interval = 2,
    
    name=log_name.replace(os.path.sep, "_"),
    config=args,
    project="results_wslao"
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.torch"))

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
        #ckpt_path = None
        if epoch % 30 == 0 or epoch == 1:
            model_name = "policy_epoch_" + str(epoch) + ".torch"
            ckpt_path = os.path.join(log_path, model_name)
            torch.save(policy.state_dict(), ckpt_path)
        return ckpt_path

    
    if not args.watch:
        # trainer
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            #save_best_fn=save_best_fn,
            #save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            resume_from_log=args.resume,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)

    policy_file_path = os.path.join(log_path, "policy.torch")
    policy.load_state_dict(torch.load(policy_file_path))

if __name__ == "__main__":
    test_ppo(get_args())
