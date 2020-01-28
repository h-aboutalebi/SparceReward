import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import os
import logging

from engine.algorithms.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


"""
From: https://github.com/pytorch/pytorch/issues/1959
There's an official LayerNorm implementation in pytorch now, but it hasn't been included in 
pip version yet. This is a temporary version
This slows down training by a bit
"""


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


nn.LayerNorm = LayerNorm


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = inputs.to(device)
        x = self.linear1(x)
        # x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = self.ln2(x)
        x = F.relu(x)
        mu = F.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.linear1(x)
        # x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)
        x = self.linear2(x)
        # x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V


class DivDDPGActor(AbstractAgent):
    def __init__(self, state_dim, action_dim, max_action,expl_noise,
                 action_high, action_low, tau, device, lr_actor,phi=0.999, linear_flag=False,hidden_size=400):
        super().__init__(state_dim, action_dim, max_action, device)
        self.number_of_time_target_policy_is_called = 0
        self.alpha = 1
        self.expl_noise = expl_noise
        self.linear_flag = linear_flag
        self.action_low = action_low
        self.action_high = action_high
        self.phi = phi
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.poly_rl_alg = None
        self.num_inputs = state_dim
        self.action_space = action_dim
        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_optim = SGD(self.actor.parameters(), lr=lr_actor, momentum=0.9)
        self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_optim = SGD(self.critic.parameters(), lr=lr_actor, momentum=0.9)
        self.gamma = 0.99
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    # This is where the behavioural policy is called
    def select_action(self, state, tensor_board_writer, step_number):
        self.actor.eval()
        state=torch.from_numpy(state).float()
        mu = self.actor((state))
        if self.expl_noise != 0:
            # import ipdb;
            # ipdb.set_trace()
            mu = mu + torch.tensor(np.random.normal(0, self.expl_noise, size=self.action_dim)
                                   .clip(min(self.action_low), max(self.action_high)),dtype=torch.float).to(self.device)
        self.actor.train()
        mu = mu.data
        return mu.clamp(-1, 1).cpu()

    # This function samples from target policy for test
    def select_action_target(self, state, tensor_board_writer=None, step_number=None):
        self.actor_target.eval()
        mu = self.actor_target(torch.Tensor(state).to(self.device))
        self.actor_target.train()
        mu = mu.data
        return mu.clamp(-1, 1).cpu()

    def train(self, replay_buffer, step_number, batch_size, writer, env_reset, delta=0.2):
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state_batch = torch.Tensor(x).to(self.device)
        action_batch = torch.Tensor(u).to(self.device)
        next_state_batch = torch.Tensor(y).to(self.device)
        mask_batch = torch.Tensor(1 - d).to(self.device)
        reward_batch = torch.Tensor(r).to(self.device)
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        pdist = nn.PairwiseDistance(p=2)
        distance_diverse = pdist(action_batch, self.actor(state_batch))
        distance_diverse = torch.clamp(distance_diverse, -delta, delta)
        distance_diverse = torch.mean(distance_diverse)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        # updating critic network
        self.critic_optim.zero_grad()
        state_action_batch = self.critic((state_batch), (action_batch))
        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # updating actor network
        self.actor_optim.zero_grad()
        if (self.linear_flag):
            self.alpha = self.alpha * self.phi
        policy_loss = -self.critic((state_batch), self.actor((state_batch)))
        policy_loss_mean = policy_loss.mean()
        policy_loss = policy_loss_mean - self.alpha * distance_diverse
        # logger.info("policy loss: {}| distance div: {}| alpha: {}".format(policy_loss_mean,distance_diverse,self.alpha))
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        # updating target policy networks with soft update
        soft_update(self.actor_target, self.actor, self.tau)
        norm_grad_actor_net = self.calculate_norm_grad(self.actor)
        soft_update(self.critic_target, self.critic, self.tau)
        return value_loss.item(), policy_loss.item()

    def calculate_norm_grad(self, net):
        S = 0
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            S += p.grad.data.norm(2).item() ** 2
        return np.sqrt(S)

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
