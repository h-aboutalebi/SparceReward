from engine.algorithms.SAC.model import QNetwork, GaussianPolicy, DeterministicPolicy
from torch.optim import Adam
from engine.algorithms.SAC.utils import soft_update, hard_update
import torch

from engine.algorithms.SAC.sac import SAC


class OAC(SAC):

    def __init__(self, state_dim, action_dim, max_action, action_space, gamma, tau, alpha, device, update_interval=1,
                 policy="Gaussian", automatic_entropy_tuning=False, hidden_size=256, lr=0.0003, start_steps=10000):
        super(OAC, self).__init__(state_dim=state_dim, action_dim=action_dim,
                                  max_action=max_action,gamma=gamma,tau=tau,
                                  alpha=alpha,device=device,
                                  policy=policy, start_steps=start_steps)
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
