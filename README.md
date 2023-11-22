# Deep Intrinsically Motivated Exploration in Continuous Control

### Published!
If you use our code and results, please cite the paper.
  ```
  @Article{Saglam2023,
      author={Saglam, Baturay
      and Kozat, Suleyman S.},
      title={Deep intrinsically motivated exploration in continuous control},
      journal={Machine Learning},
      year={2023},
      month={Oct},
      day={26},
      issn={1573-0565},
      doi={10.1007/s10994-023-06363-4},
      url={https://doi.org/10.1007/s10994-023-06363-4}
  }  
  ```  


#
PyTorch implementation of the [_Deep Intrinsically Motivated Exploration_ algorithm (DISCOVER)](https://doi.org/10.1007/s10994-023-06363-4). 
Note that the implementation of the baseline algorithms are heavily based on the following repositories:

- [DDPG](https://arxiv.org/abs/1509.02971) and [TD3](https://arxiv.org/abs/1802.09477): https://github.com/sfujim/TD3
- [A2C](https://arxiv.org/abs/1602.01783) and [PPO](https://arxiv.org/abs/1707.06347): https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- [SAC](https://arxiv.org/abs/1801.01290): Our implementation

[DDPG](https://arxiv.org/abs/1509.02971) uses the baseline hyper-parameters as outlined in the original [paper](https://arxiv.org/abs/1509.02971).
[TD3](https://arxiv.org/abs/1802.09477) is the fine-tuned version imported from the [author's Pytorch implementation of the TD3 algorithm](https://github.com/sfujim/TD3). 
Tuned hyper-parameters for [SAC](https://arxiv.org/abs/1801.01290) are imported from [OpenAI Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), 
and [A2C](https://arxiv.org/abs/1602.01783) and [PPO](https://arxiv.org/abs/1707.06347) follow the [repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) for the tuned hyper-parameters for the [OpenAI Gym](https://gym.openai.com/) continuous control tasks. 

The algorithm is tested on [MuJoCo](https://gym.openai.com/envs/#mujoco) and [Box2D](https://gym.openai.com/envs/#box2d) continuous control tasks.

### Results
Learning curves are found under [./Learning Curves](https://github.com/baturaysaglam/DISCOVER/tree/main/Learning%20Curves). 
Corresponding learning figures are found under [./Learning Figures](https://github.com/baturaysaglam/DISCOVER/tree/main/Learning%20Figures). 
Each learning curve is formatted as NumPy arrays of 1001 evaluations (1001,). 
Each evaluation corresponds to the average reward from running the policy for 10 episodes without exploration and updates. 
The randomly initialized policy network produces the first evaluation. Evaluations are performed every 1000 time steps, over 1 million time steps for 10 random seeds.

### Computing Infrastructure
Following computing infrastructure is used to produce the results.
| Hardware/Software  | Model/Version |
| ------------- | ------------- |
| Operating System  | Ubuntu 18.04.5 LTS  |
| CPU  | AMD Ryzen 7 3700X 8-Core Processor |
| GPU  | Nvidia GeForce RTX 2070 SUPER |
| CUDA  | 11.1  |
| Python  | 3.8.5 |
| PyTorch  | 1.8.1 |
| OpenAI Gym  | 0.17.3 |
| MuJoCo  | 1.50 |
| Box2D  | 2.3.10 |
| NumPy  | 1.19.4 |

### Usage - DDPG & TD3
```
usage: main.py [-h] [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--start_time_steps N] [--buffer_size BUFFER_SIZE]
               [--eval_freq N] [--max_time_steps N]
               [--exp_regularization EXP_REGULARIZATION]
               [--exploration_noise G] [--batch_size N] [--discount G]
               [--tau G] [--policy_noise G] [--noise_clip G] [--policy_freq N]
               [--save_model] [--load_model LOAD_MODEL]
```

### Arguments - DDPG & TD3
```
DDPG, TD3 and their DISCOVER Implementation

optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: DISCOVER_TD3)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                        (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --start_time_steps N  Number of exploration time steps sampling random
                        actions (default: 1000)
  --buffer_size BUFFER_SIZE
                        Size of the experience replay buffer (default:
                        1000000)
  --eval_freq N         Evaluation period in number of time steps (default:
                        1000)
  --max_time_steps N    Maximum number of steps (default: 1000000)
  --exp_regularization EXP_REGULARIZATION
  --exploration_noise G
                        Std of Gaussian exploration noise
  --batch_size N        Batch size (default: 256)
  --discount G          Discount factor for reward (default: 0.99)
  --tau G               Learning rate in soft/hard updates of the target
                        networks (default: 0.005)
  --policy_noise G      Noise added to target policy during critic update
  --noise_clip G        Range to clip target policy noise
  --policy_freq N       Frequency of delayed policy updates
  --save_model          Save model and optimizer parameters
  --load_model LOAD_MODEL
                        Model load file name; if empty, does not load
  ```

### Usage - SAC
```
usage: main.py [-h] [--policy POLICY] [--policy_type POLICY_TYPE] [--env ENV]
               [--seed SEED] [--gpu GPU] [--start_steps N]
               [--exp_regularization EXP_REGULARIZATION]
               [--buffer_size BUFFER_SIZE] [--eval_freq N] [--num_steps N]
               [--batch_size N] [--hard_update G] [--train_freq N]
               [--updates_per_step N] [--target_update_interval N] [--alpha G]
               [--automatic_entropy_tuning G] [--reward_scale N] [--gamma G]
               [--tau G] [--lr G] [--hidden_size N]
```

### Arguments - SAC
```
SAC and its DISCOVER Implementation

optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: SAC)
  --policy_type POLICY_TYPE
                        Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                        (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --start_steps N       Number of exploration time steps sampling random
                        actions (default: 1000)
  --exp_regularization EXP_REGULARIZATION
  --buffer_size BUFFER_SIZE
                        Size of the experience replay buffer (default:
                        1000000)
  --eval_freq N         evaluation period in number of time steps (default:
                        1000)
  --num_steps N         Maximum number of steps (default: 1000000)
  --batch_size N        Batch size (default: 256)
  --hard_update G       Hard update the target networks (default: True)
  --train_freq N        Frequency of the training (default: 1)
  --updates_per_step N  Model updates per training time step (default: 1)
  --target_update_interval N
                        Number of critic function updates per training time
                        step (default: 1)
  --alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  --automatic_entropy_tuning G
                        Automatically adjust α (default: False)
  --reward_scale N      Scale of the environment rewards (default: 5)
  --gamma G             Discount factor for reward (default: 0.99)
  --tau G               Learning rate in soft/hard updates of the target
                        networks (default: 0.005)
  --lr G                Learning rate (default: 0.0003)
  --hidden_size N       Hidden unit size in neural networks (default: 256)
  ```

### Usage - A2C
```
usage: main.py [-h] [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--eval_freq N] [--max_time_steps N]
               [--exp_regularization EXP_REGULARIZATION]
               [--learning_rate LEARNING_RATE] [--adam_eps ADAM_EPS]
               [--alpha ALPHA] [--gamma G] [--use_gae USE_GAE]
               [--recurrent_policy RECURRENT_POLICY] [--gae_lambda GAE_LAMBDA]
               [--entropy_coef ENTROPY_COEF]
               [--value_loss_coef VALUE_LOSS_COEF]
               [--max_grad_norm MAX_GRAD_NORM]
               [--n_rollout_steps N_ROLLOUT_STEPS]
               [--use_linear_lr_decay USE_LINEAR_LR_DECAY]
               [--use_proper_time_limits USE_PROPER_TIME_LIMITS]
               [--cuda_deterministic] [--num_processes NUM_PROCESSES]
               [--save_model] [--load_model LOAD_MODEL]
```

### Arguments - A2C
```
A2C and its DISCOVER Implementation

optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: DISCOVER_A2C)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                        (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --eval_freq N         Evaluation period in number of time steps (default:
                        1000)
  --max_time_steps N    Maximum number of steps (default: 1000000)
  --exp_regularization EXP_REGULARIZATION
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.0013)
  --adam_eps ADAM_EPS   Optimizer epsilon (default: 1e-5)
  --alpha ALPHA         Optimizer alpha (default: 0.99)
  --gamma G             Discount factor of rewards (default: 0.99)
  --use_gae USE_GAE     Usage of Generalized Advantage Estimation
  --recurrent_policy RECURRENT_POLICY
                        Usage of recurrent policy (Default: False)
  --gae_lambda GAE_LAMBDA
                        GAE lambda parameter (default: 0.95)
  --entropy_coef ENTROPY_COEF
                        Entropy term coefficient (default: 0.0)
  --value_loss_coef VALUE_LOSS_COEF
                        Value loss coefficient (default: 0.5)
  --max_grad_norm MAX_GRAD_NORM
                        Maximum norm of gradients (default: 0.5)
  --n_rollout_steps N_ROLLOUT_STEPS
                        Number of rollout steps (default: 5)
  --use_linear_lr_decay USE_LINEAR_LR_DECAY
                        Usage of a linear schedule on the learning rate
  --use_proper_time_limits USE_PROPER_TIME_LIMITS
                        Compute returns taking into account time limits
  --cuda_deterministic  Sets flags for determinism when using CUDA
                        (potentially slow!)
  --num_processes NUM_PROCESSES
                        How many training CPU processes to use (default: 1)
  --save_model          Save model and optimizer parameters
  --load_model LOAD_MODEL
                        Model load file name; if empty, does not load
  ```

### Usage - PPO
```
usage: main.py [-h] [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--eval_freq N] [--max_time_steps N]
               [--exp_regularization EXP_REGULARIZATION]
               [--learning_rate LEARNING_RATE] [--adam_eps ADAM_EPS]
               [--alpha ALPHA] [--gamma G] [--use_gae USE_GAE]
               [--recurrent_policy RECURRENT_POLICY] [--gae_lambda GAE_LAMBDA]
               [--entropy_coef ENTROPY_COEF]
               [--value_loss_coef VALUE_LOSS_COEF]
               [--max_grad_norm MAX_GRAD_NORM]
               [--n_rollout_steps N_ROLLOUT_STEPS] [--n_epochs N_EPOCHS]
               [--num_mini_batch NUM_MINI_BATCH] [--clip_param CLIP_PARAM]
               [--use_linear_lr_decay USE_LINEAR_LR_DECAY]
               [--use_proper_time_limits USE_PROPER_TIME_LIMITS]
               [--cuda_deterministic] [--num_processes NUM_PROCESSES]
               [--save_model] [--load_model LOAD_MODEL]
```

### Arguments - PPO
```
PPO and its DISCOVER Implementation

optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: DISCOVER_PPO)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                        (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --eval_freq N         Evaluation period in number of time steps (default:
                        1000)
  --max_time_steps N    Maximum number of steps (default: 1000000)
  --exp_regularization EXP_REGULARIZATION
  --learning_rate LEARNING_RATE
                        Learning rate (default: 3e-4)
  --adam_eps ADAM_EPS   Optimizer epsilon (default: 1e-5)
  --alpha ALPHA         Optimizer alpha (default: 0.99)
  --gamma G             Discount factor of rewards (default: 0.99)
  --use_gae USE_GAE     Usage of Generalized Advantage Estimation
  --recurrent_policy RECURRENT_POLICY
                        Usage of recurrent policy (Default: False)
  --gae_lambda GAE_LAMBDA
                        GAE lambda parameter (default: 0.95)
  --entropy_coef ENTROPY_COEF
                        Entropy term coefficient (default: 0.0)
  --value_loss_coef VALUE_LOSS_COEF
                        Value loss coefficient (default: 0.5)
  --max_grad_norm MAX_GRAD_NORM
                        Maximum norm of gradients (default: 0.5)
  --n_rollout_steps N_ROLLOUT_STEPS
                        Number of rollout steps (default: 2048)
  --n_epochs N_EPOCHS   Number of epochs (default: 10)
  --num_mini_batch NUM_MINI_BATCH
                        Number of mini-batches (default: 32)
  --clip_param CLIP_PARAM
                        Clip parameter (default: 0.2)
  --use_linear_lr_decay USE_LINEAR_LR_DECAY
                        Usage of a linear schedule on the learning rate
  --use_proper_time_limits USE_PROPER_TIME_LIMITS
                        Compute returns taking into account time limits
  --cuda_deterministic  Sets flags for determinism when using CUDA
                        (potentially slow!)
  --num_processes NUM_PROCESSES
                        How many training CPU processes to use (default: 1)
  --save_model          Save model and optimizer parameters
  --load_model LOAD_MODEL
                        Model load file name; if empty, does not load
  ```
