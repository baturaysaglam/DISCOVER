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


class DISCOVER_PPO():
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 exp_regularization,
                 policy,
                 clip_param,
                 n_epochs,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 learning_rate=None,
                 adam_eps=None,
                 max_grad_norm=None,
                 device=None,
                 use_clipped_value_loss=True):

        self.policy = policy
        self.explorer = Explorer(state_dim, max_action, exp_regularization).to(device)

        self.clip_param = clip_param
        self.n_epochs = n_epochs
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=adam_eps)
        self.explorer_optimizer = optim.Adam(self.explorer.parameters(), lr=learning_rate, eps=adam_eps)

    def explore(self, inputs):
        return self.explorer(inputs)

    def update_parameters(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.n_epochs):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, noise_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                    obs_batch, noise_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                # Compute the clipped surrogate (policy) loss
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

                action_loss = -torch.min(surr1, surr2).mean()

                # Compute the value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Optimize the policy
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Compute the explorer loss
                values, _, _, _ = self.policy.evaluate_actions(
                    obs_batch, self.explorer(obs_batch), recurrent_hidden_states_batch, masks_batch, actions_batch)

                # Compute the value loss for the explorer policy gradient
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Optimize the explorer policy
                self.explorer_optimizer.zero_grad()
                (-value_loss * self.value_loss_coef).backward()
                nn.utils.clip_grad_norm_(self.explorer.parameters(), self.max_grad_norm)
                self.explorer_optimizer.step()
