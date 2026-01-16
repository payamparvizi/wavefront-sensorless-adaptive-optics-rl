import argparse
import torch


def rounded_float(x):
    return round(float(x), 3)  # keep 3 decimal places
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='AO-v0')
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--buffer_size", type=int, default=15000)
    # parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--hidden_size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epoch", type=int, default=1200)
    parser.add_argument("--step_per_epoch", type=int, default=3200)
    parser.add_argument("--step_per_collect", type=int, default=400)
    parser.add_argument("--repeat-per-collect", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--training_num", type=int, default=50)
    parser.add_argument("--test-num", type=int, default=50)

    parser.add_argument("--rew-norm", type=int, default=1)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=False)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps_clip", type=float, default=0.05)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument('--episodes_per_test', type=int, default=20, help="How many episodes to run for evaluating the policy during training.")
    parser.add_argument('--deterministic_eval', default=1, type=int, choices=[0,1], help="Use deterministic action evaluation during testing or stochastic?")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--saved_policy", dest='saved_policy', type=int, default=0)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="results_cc")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    
    parser.add_argument("--regularization_case", type=str, default="standard_PPO")
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--stack_num', type=int, default=7)
    # parser.add_argument('--state_with_action', type=bool, default=1)
    parser.add_argument("--save_interval", type=int, default=4)
    
    # Lipsnet parameters:
    parser.add_argument("--global_lips", type=int, default=1)                  # 1: LispNet-G, 0: LipsNet-L
    parser.add_argument("--k_init", type=float, default=0.3)                     # Initial Lipschitz constant K_init
    parser.add_argument("--loss_lambda", type=float, default=1e-7)             # weight lambda
    parser.add_argument("--eps_k_net", type=float, default=1e-4)               # Small constant epsilon
    parser.add_argument("--squash_action", type=int, default=1)
    parser.add_argument("--lr_k_net", type=float, default=1e-7)                 # Actor learning rate ηk
    parser.add_argument("--k_sizes", type=int, default=192)
    
    # CAPS parameters
    parser.add_argument("--lambda_T", type=float, default=0.5)
    parser.add_argument("--lambda_S", type=float, default=0.05)
    parser.add_argument("--sigma_s_bar", type=float, default=100)
    
    # APS patameters
    parser.add_argument("--lambda_P", type=rounded_float, default=0.075)
    parser.add_argument("--c_homog", type=rounded_float, default=3.0)
    parser.add_argument("--noise_pym", type=float, default=1e-12)
    
    # Environment-Specific parameters
    parser.add_argument('--atm_type', dest='atm_type', default='dynamic', choices=['quasi_static', 'semi_dynamic', 'dynamic'])
    parser.add_argument('--atm_vel', type=float, default=50)
    parser.add_argument('--atm_fried', type=float, default=0.10)
    parser.add_argument('--act_type', default='zernike', choices=['zernike', 'num_actuators'])
    parser.add_argument('--act_dim', type=int, default=9)
    parser.add_argument('--obs_dim', type=int, default=5)
    parser.add_argument('--rew_type', default='smf_ssim', choices=['strehl_ratio', 'smf_ssim'])
    parser.add_argument('--reward_threshold', type=float, default=None)
    parser.add_argument("--flat_mirror_start_per_episode", type=int, default=0)
    parser.add_argument("--SH_operation", dest='SH_operation', type=int, default=0)    # 0 -> RL, 1 -> flat, 2 -> shack
    parser.add_argument('--delta_t', type=float, default=1e-4)
    parser.add_argument('--c_act_range', dest='c_act_range', type=int, default=10)
    parser.add_argument('--c_rand', dest='c_rand', type=int, default=1)
    parser.add_argument('--c_mult', dest='c_mult', type=float, default=0.2)
    parser.add_argument('--c_rew', dest='c_rew', type=float, default=1)
    parser.add_argument('--c_mode1', dest='c_mode1', type=float, default=1.0)

    return parser.parse_args()
