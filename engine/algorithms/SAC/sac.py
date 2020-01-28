import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from engine.algorithms.abstract_agent import AbstractAgent
from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, DeterministicPolicy


# From: https://github.com/pranz24/pytorch-soft-actor-critic


class SAC(AbstractAgent):
    def __init__(self, state_dim, action_dim, max_action, action_space, gamma, tau, alpha, device, update_interval=1,
                 policy="Gaussian", automatic_entropy_tuning=False, hidden_size=256, lr=0.0003, start_steps=10000):
        super(SAC, self).__init__(state_dim=state_dim, action_dim=action_dim,
                                  max_action=max_action, device=device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.start_steps = start_steps
        self.counter_actions = 0
        self.policy_type = policy
        self.update_interval = update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.critic = QNetwork(state_dim, action_dim, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.action_space = action_space

        self.critic_target = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(state_dim, action_dim, hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(state_dim, action_space.shape[0], hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, tensor_board_writer=None, step_number=None):
        self.counter_actions += 1
        state = torch.Tensor(state).reshape(1, -1)
        action, _, _ = self.policy.sample(state.to(self.device))
        if (self.start_steps < self.counter_actions):
            return action.detach().cpu().numpy()[0]
        else:
            return self.action_space.sample()

    def select_action_target(self, state, tensor_board_writer=None, step_number=None,previous_action=None):
        state = torch.Tensor(state).reshape(1, -1)
        _, _, action = self.policy.sample(state.to(self.device))
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size, step_number, writer=None, env_reset=None):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(
            batch_size=batch_size)

        state_batch = torch.Tensor(state_batch).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.Tensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((
                               self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if step_number % self.update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
