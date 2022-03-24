import argparse
from collections import deque
import os
import socket
import time

import numpy as np
import torch

from PPO import PPO
from DISCOVER_PPO import DISCOVER_PPO
from env import make_vec_envs
from model import Policy as PPOPolicy
from discover_model import Policy as DISCOVERPolicy
from storage import RolloutStorage, RolloutStorageExploration
from evaluation import evaluate
from utils import update_linear_schedule, get_vec_normalize


# PPO tuned MuJoCo hyper-parameters are imported from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO and its DISCOVER Implementation')
    parser.add_argument("--policy", default="DISCOVER_PPO", help='Algorithm (default: DISCOVER_PPO)')
    parser.add_argument("--env", default="Walker2d-v2", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=0, type=int,
                        help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--eval_freq", default=1e3, metavar='N', type=int,
                        help='Evaluation period in number of time steps (default: 1000)')
    parser.add_argument("--max_time_steps", default=1000000, type=int, metavar='N',
                        help='Maximum number of steps (default: 1000000)')
    parser.add_argument("--exp_regularization", default=0.1, type=float)
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--adam_eps', type=float, default=1e-5, help='Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='Optimizer alpha (default: 0.99)')
    parser.add_argument("--gamma", default=0.99, metavar='G', help='Discount factor of rewards (default: 0.99)')
    parser.add_argument('--use_gae', default=True, help='Usage of Generalized Advantage Estimation')
    parser.add_argument('--recurrent_policy', default=False, help='Usage of recurrent policy (Default: False)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.0, help='Entropy term coefficient (default: 0.0)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum norm of gradients (default: 0.5)')
    parser.add_argument('--n_rollout_steps', type=int, default=2048, help='Number of rollout steps (default: 2048)')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--num_mini_batch', type=int, default=32, help='Number of mini-batches (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2, help='Clip parameter (default: 0.2)')
    parser.add_argument('--use_linear_lr_decay', default=True,
                        help='Usage of a linear schedule on the learning rate')
    parser.add_argument('--use_proper_time_limits', default=True,
                        help='Compute returns taking into account time limits')
    parser.add_argument('--cuda_deterministic', action='store_true', default=False,
                        help="Sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num_processes', type=int, default=1,
                        help='How many training CPU processes to use (default: 1)')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(f"./results"):
        os.makedirs(f"./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    envs = make_vec_envs(args.env, args.seed, args.num_processes, args.gamma, device, False)

    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    max_action = float(envs.action_space.high[0])

    if "DISCOVER" in args.policy:
        policy = DISCOVERPolicy(envs.observation_space.shape, envs.action_space, device=device, base_kwargs={'recurrent': args.recurrent_policy})
    else:
        policy = PPOPolicy(envs.observation_space.shape, envs.action_space, device=device, base_kwargs={'recurrent': args.recurrent_policy})

    policy.to(device)

    agent_kwargs = {
        "policy": policy,
        "clip_param": args.clip_param,
        "n_epochs": args.n_epochs,
        "num_mini_batch": args.num_mini_batch,
        "value_loss_coef": args.value_loss_coef,
        "entropy_coef": args.entropy_coef,
        "learning_rate": args.learning_rate,
        "adam_eps": args.adam_eps,
        "max_grad_norm": args.max_grad_norm
    }

    rollout_kwargs = {
        "num_steps": args.n_rollout_steps,
        "num_processes": args.num_processes,
        "obs_shape": envs.observation_space.shape,
        "action_space": envs.action_space,
        "recurrent_hidden_state_size": policy.recurrent_hidden_state_size
    }

    discover = False

    if args.policy == 'PPO':
        agent = PPO(**agent_kwargs)
        rollouts = RolloutStorage(**rollout_kwargs)
    elif args.policy == 'DISCOVER_PPO':
        agent = DISCOVER_PPO(state_dim=state_dim,
                             action_dim=action_dim,
                             max_action=max_action,
                             exp_regularization=args.exp_regularization,
                             device=device,
                             **agent_kwargs)
        rollouts = RolloutStorageExploration(**rollout_kwargs)
        discover = True

        if args.recurrent_policy:
            raise NotImplementedError("DISCOVER is not implemented for recurrent policies")

    evaluations = [f"HOST: {socket.gethostname()}", f"GPU: {torch.cuda.get_device_name(args.gpu)}"]

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()

    num_updates = int(args.max_time_steps) // args.n_rollout_steps // args.num_processes
    total_training_time_steps = 0

    for update_idx in range(num_updates):
        if args.use_linear_lr_decay:
            # Linearly decrease the learning rate per update
            update_linear_schedule(agent.optimizer, update_idx, num_updates, args.learning_rate)

            if discover:
                update_linear_schedule(agent.explorer_optimizer, update_idx, num_updates, args.learning_rate)

        for step in range(args.n_rollout_steps):
            # Sample actions from the stochastic policy
            with torch.no_grad():
                if discover:
                    exploration_direction = agent.explore(rollouts.obs[step])

                    value, action, action_log_prob, recurrent_hidden_states = policy.act(
                        rollouts.obs[step], exploration_direction, rollouts.recurrent_hidden_states[step], rollouts.masks[step])
                else:
                    value, action, action_log_prob, recurrent_hidden_states = policy.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            # Receive the reward and observe the next state
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done, then clean the history of observations
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            if discover:
                rollouts.insert(obs, recurrent_hidden_states, action, exploration_direction, action_log_prob, value, reward, masks, bad_masks)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

            if total_training_time_steps % args.eval_freq == 0 and len(episode_rewards) > 1:
                obs_rms = get_vec_normalize(envs).obs_rms

                evaluation_reward = evaluate(policy, obs_rms, args.env, args.seed, args.num_processes, device, discover)

                evaluations.append(evaluation_reward)
                np.save(f"./results/{file_name}", evaluations)

            total_training_time_steps += 1

        if discover:
            with torch.no_grad():
                next_value = policy.get_value(rollouts.obs[-1], exploration_direction, rollouts.recurrent_hidden_states[-1],
                                              rollouts.masks[-1]).detach()
        else:
            with torch.no_grad():
                next_value = policy.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                              rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        agent.update_parameters(rollouts)

        rollouts.after_update()

        if len(episode_rewards) > 1:
            total_n_rollout_steps = (update_idx + 1) * args.num_processes * args.n_rollout_steps

            end = time.time()

            print(f"Total updates: {update_idx + 1} Total Time Steps: {total_n_rollout_steps} FPS: {int(total_n_rollout_steps / (end - start))} "
                  f"Last 10 Training Episodes Reward: mean/median {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}, "
                  f"max/min {np.max(episode_rewards):.1f}/{np.min(episode_rewards):.1f}")
