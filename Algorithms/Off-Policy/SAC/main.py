import argparse
import itertools
import os
import socket

import gym
import numpy as np
import torch

from SAC import SAC
from DISCOVER_SAC import DISCOVER_SAC
from utils import ExperienceReplayBuffer, NoisyExperienceReplayBuffer


# SAC tuned hyper-parameters are imported from
# OpenAI Baselines3 Zoo: https://github.com/DLR-RM/rl-baselines3-zoo
# Original paper: https://arxiv.org/abs/1801.01290


def hyper_parameter_dict(args):
    if args.env == "BipedalWalker-v3":
        args.gamma = 0.98
        args.tau = 0.02
        args.train_freq = 64
        args.updates_per_step = 64
    elif args.env == "Hopper-v2" or args.env == "Swimmer-v2" or args.env == "LunarLanderContinuous-v2":
        args.updates_per_step = 1

        if args.env == "Swimmer-v2":
            args.gamma = 0.9999
        elif args.env == "LunarLanderContinuous-v2":
            args.tau = 0.01
    elif args.env == "HumanoidStandup-v2":
        args.reward_scale = 100
        args.alpha = 0.05
    elif args.env == "Humanoid-v2":
        args.reward_scale = 20
        args.alpha = 0.05

    if args.hard_update:
        args.tau = 1
        args.target_update_interval = 1000

        if args.env != "Humanoid-v2" and args.env != "HumanoidStandup-v2" and args.env != "Hopper-v2":
            args.updates_per_step = 4

    if args.env == "Ant-v2" or args.env == "HalfCheetah-v2" or args.env == "LunarLanderContinuous-v2" or args.env == "BipedalWalker-v3":
        args.start_steps = 10000

    return args


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_policy(agent, env_name, seed, discover, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    episodes = 10

    for _ in range(eval_episodes):
        state = eval_env.reset()
        episode_reward = 0
        done = False

        while not done:
            if discover:
                action, _ = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state, evaluate=True)

            next_state, reward, done, _ = eval_env.step(action)
            episode_reward += reward

            state = next_state

        avg_reward += episode_reward

    avg_reward /= episodes

    print("---------------------------------------")
    print(f"Evaluation over {episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward


parser = argparse.ArgumentParser(description='SAC and its DISCOVER Implementation')

parser.add_argument('--policy', default="DISCOVER_SAC", help='Algorithm (default: DISCOVER_SAC)')
parser.add_argument('--policy_type', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--env', default="Walker2d-v2", help='OpenAI Gym environment name')
parser.add_argument('--seed', type=int, default=0, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)')
parser.add_argument('--gpu', type=int, default=0, help='GPU ordinal for multi-GPU computers (default: 0)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Number of exploration time steps sampling random actions (default: 1000)')
parser.add_argument("--exp_regularization", default=0.3, type=float)
parser.add_argument('--buffer_size', type=int, default=1000000, help='Size of the experience replay buffer (default: '
                                                                     '1000000)')
parser.add_argument('--eval_freq', type=int, default=1000, metavar='N', help='evaluation period in number of time '
                                                                             'steps (default: 1000)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='Maximum number of steps (default: 1000000)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='Batch size (default: 256)')
parser.add_argument('--hard_update', type=bool, default=True, metavar='G', help='Hard update the target networks ('
                                                                                'default: True)')
parser.add_argument('--train_freq', type=int, default=1, metavar='N',
                    help='Frequency of the training (default: 1)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='Model updates per training time step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Number of critic function updates per training time step (default: 1)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--reward_scale', type=float, default=5.0, metavar='N', help='Scale of the environment rewards ('
                                                                                 'default: 5)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='Learning rate in soft/hard updates of the target networks (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='Learning rate (default: 0.0003)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='Hidden unit size in neural networks (default: 256)')

args = parser.parse_args()

# Adjust the hyper-parameters with respect to the environment
args = hyper_parameter_dict(args)

# Target update specific parameters
print(f"\nEnvironment: {args.env}\n")

print(f"Policy type: {args.policy_type}\n")

if args.hard_update:
    print(f"Update: HARD\n")
else:
    print(f"Update: SOFT\n")

print(f"Tau: {args.tau}")
print(f"Target update interval: {args.target_update_interval}")
print(f"Updates per step: {args.updates_per_step}\n")

print(f"Reward scale: {args.reward_scale}\n")
print(f"Start time steps: {args.start_steps}\n")

file_name = f"{args.policy}_{args.env}_{args.seed}"

if not os.path.exists("./results"):
    os.makedirs("./results")

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Environment
env = gym.make(args.env)
env.seed(args.seed)
env.action_space.seed(args.seed)
max_action = float(env.action_space.high[0])

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
if args.policy == "SAC":
    agent = SAC(env.observation_space.shape[0], env.action_space, args, device)
    replay_buffer = ExperienceReplayBuffer(args.buffer_size, args.seed)
    discover = False
elif args.policy == "DISCOVER_SAC":
    agent = DISCOVER_SAC(env.observation_space.shape[0], env.action_space, max_action, args, device)
    replay_buffer = NoisyExperienceReplayBuffer(args.buffer_size, args.seed)
    discover = True

print("---------------------------------------")
print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
print("---------------------------------------")

# Training Loop
total_numsteps = 0
updates = 0

# Evaluate untrained policy
evaluations = [f"HOST: {socket.gethostname()}", f"GPU: {torch.cuda.get_device_name(args.gpu)}",
               evaluate_policy(agent, args.env, args.seed, discover)]

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()

            if discover:
                _, exploration = agent.select_action(state)
            else:
                exploration = np.zeros_like(action)
        else:
            if discover:
                action, exploration = agent.select_action(state, evaluate=False)
            else:
                action = agent.select_action(state, evaluate=False)
                exploration = np.random.normal(0, max_action * args.exp_regularization, size=env.action_space.shape[0])

        action = (action + exploration).clip(-max_action, max_action)

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        reward *= args.reward_scale

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        if discover:
            replay_buffer.add(state, action, exploration, reward, next_state, mask)
        else:
            replay_buffer.add(state, action, reward, next_state, mask)

        state = next_state

        if total_numsteps >= args.start_steps:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                agent.update_parameters(replay_buffer, args.batch_size, updates)
                updates += 1

        if total_numsteps % args.eval_freq == 0:
            evaluations.append(evaluate_policy(agent, args.env, args.seed, discover))
            np.save(f"./results/{file_name}", evaluations)

    if total_numsteps > args.num_steps:
        break

    print(f"Total T: {total_numsteps} Episode Num: {i_episode} Episode T: "
          f"{episode_steps}" f" Reward: {episode_reward:.3f}")
