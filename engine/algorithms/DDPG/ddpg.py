import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from engine.algorithms.abstract_agent import AbstractAgent


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

# Scott Fujimoto implementation

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3(x)
        return x


class DDPG(AbstractAgent):
    def __init__(self, state_dim, action_dim, max_action, expl_noise, action_high, action_low, tau,device,lr_actor):
        super(DDPG, self).__init__(state_dim=state_dim, action_dim=action_dim,
                                   max_action=max_action,device=device)

        self.expl_noise = expl_noise
        self.action_dim = action_dim
        self.action_high = action_high
        self.tau = tau
        self.action_low = action_low
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

    def select_action(self, state, tensor_board_writer=None, previous_action=None, step_number=None):
        state = np.array(state)
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if self.expl_noise != 0:
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(
                self.action_low, self.action_high)
        return action

    def select_action_target(self, state,  tensor_board_writer=None, step_number=None):
        # print(state)
        state = np.array(state)
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor_target(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, step_number, batch_size=64, gamma=0.99, writer=None, **kwargs):

        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.Tensor(x).to(self.device)
        action = torch.Tensor(u).to(self.device)
        next_state = torch.Tensor(y).to(self.device)
        done = torch.Tensor(1 - d).to(self.device)
        reward = torch.Tensor(r).to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (done * gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
