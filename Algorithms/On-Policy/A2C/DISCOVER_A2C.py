import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import init


class Explorer(nn.Module):
    def __init__(self, state_dim, max_action, exp_regularization):
        super(Explorer, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.l1 = init_(nn.Linear(state_dim, 64))
        self.l2 = init_(nn.Linear(64, 64))
        self.l3 = init_(nn.Linear(64, state_dim))

        self.max_action = max_action
        self.exp_regularization = exp_regularization

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        return self.max_action * torch.tanh(self.l3(a)) * self.exp_regularization ** 2


class DISCOVER_A2C():
    def __init__(self,
                 state_dim,
                 max_action,
                 exp_regularization,
                 policy,
                 value_loss_coef,
                 entropy_coef,
                 learning_rate=None,
                 adam_eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 device=None):
        self.policy = policy
        self.explorer = Explorer(state_dim, max_action, exp_regularization).to(device)

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(policy.parameters(), learning_rate, eps=adam_eps, alpha=alpha)
        self.explorer_optimizer = optim.Adam(self.explorer.parameters(), lr=learning_rate, eps=adam_eps)

    def explore(self, inputs):
        return self.explorer(inputs)

    def update_parameters(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.exploration_directions.view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.policy.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Compute the explorer loss
        values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            self.explorer(rollouts.obs[:-1].view(-1, *obs_shape)),
            rollouts.recurrent_hidden_states[0].view(-1, self.policy.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        self.explorer_optimizer.zero_grad()
        (-value_loss * self.value_loss_coef).backward()
        nn.utils.clip_grad_norm_(self.explorer.parameters(), self.max_grad_norm)
        self.explorer_optimizer.step()

