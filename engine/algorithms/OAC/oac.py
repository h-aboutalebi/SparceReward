from engine.algorithms.OAC.optimistic_exploration import get_optimistic_exploration_action
from engine.algorithms.SAC.model import QNetwork, GaussianPolicy, DeterministicPolicy
from torch.optim import Adam
from engine.algorithms.SAC.utils import soft_update, hard_update
import torch

from engine.algorithms.SAC.sac import SAC


class OAC(SAC):

    def __init__(self, state_dim, action_dim, max_action, action_space, gamma, tau, alpha, device, beta_UB,delta,update_interval=1,
                 policy="Gaussian", automatic_entropy_tuning=False, hidden_size=256, lr=0.0003, start_steps=10000):
        super(OAC, self).__init__(state_dim, action_dim, max_action, action_space=action_space,
                                  gamma=gamma, alpha=alpha, tau=tau, policy=policy, device=device, start_steps=start_steps,
                                  lr=lr, automatic_entropy_tuning=automatic_entropy_tuning, hidden_size=hidden_size,
                                  update_interval=update_interval)
        self.beta_UB=beta_UB
        self.delta=delta

    def select_action(self, state, tensor_board_writer=None, step_number=None):
        hyper_params={'beta_UB':self.beta_UB,
                      'delta':self.delta}
        mean, log_std = self.policy(state)
        std=log_std.exp()
        self.counter_actions += 1
        state = torch.Tensor(state).reshape(1, -1)
        action, _, _ = self.policy.sample(state.to(self.device))
        if (self.start_steps < self.counter_actions):
            return action.detach().cpu().numpy()[0]
        else:
            return get_optimistic_exploration_action(ob_np=state, std=std,mean=mean)
