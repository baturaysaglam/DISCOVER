import torch.nn as nn
import torch.optim as optim


# Implementation of the Advantage Actor-Critic (A2C) algorithm
# Paper: https://arxiv.org/abs/1602.01783
# Note: This implementation heavily relies on the repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

class A2C():
    def __init__(self,
                 policy,
                 value_loss_coef,
                 entropy_coef,
                 learning_rate=None,
                 adam_eps=None,
                 alpha=None,
                 max_grad_norm=None):
        self.policy = policy

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(policy.parameters(), learning_rate, eps=adam_eps, alpha=alpha)

    def update_parameters(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
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

        return value_loss.item(), action_loss.item(), dist_entropy.item()
