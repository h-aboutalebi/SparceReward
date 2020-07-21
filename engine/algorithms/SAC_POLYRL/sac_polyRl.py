import torch
import numpy as np

from engine.algorithms.DDPG_POLYRL.poly_rl import PolyRL
from engine.algorithms.SAC.sac import SAC


class SAC_Poly_RL(SAC):

    def __init__(self, state_dim, action_dim, max_action, action_space, gamma, tau, alpha, device,
                 betta, epsilon, sigma_squared, lambda_, nb_actions, min_action, nb_observations, update_interval=1,
                 policy="Gaussian", automatic_entropy_tuning=False, hidden_size=256, lr=0.0003, start_steps=10000):
        super().__init__(state_dim, action_dim, max_action, action_space, gamma, tau, alpha, device,
                         update_interval, policy, automatic_entropy_tuning, hidden_size, lr, start_steps)
        self.poly_rl_alg = PolyRL(gamma=gamma, betta=betta, epsilon=epsilon, sigma_squared=sigma_squared,
                                  actor_target_function=self.select_action_target, lambda_=lambda_, nb_actions=nb_actions,
                                  nb_observations=nb_observations, max_action=max_action, min_action=min_action)
        self.nb_environment_reset=0
        self.previous_state = None

    def get_exploration_percentage(self):
        return self.poly_rl_alg.percentage_exploration

    def mod_select_action(self, state, tensor_board_writer, previous_action, step_number, nb_environment_reset):
        self.counter_actions += 1
        if (self.start_steps < self.counter_actions):
            state = torch.Tensor(state).reshape(1, -1)
            action, _, _ = self.policy.sample(state.to(self.device))
            return action.detach().cpu().numpy()[0]
        else:
            # print(self.counter_actions)
            state = np.array(state)
            if (nb_environment_reset > self.nb_environment_reset):
                self.nb_environment_reset = nb_environment_reset
                self.previous_state = None
                self.poly_rl_alg.reset_parameters_in_beginning_of_episode(self.nb_environment_reset)
            self.nb_environment_reset = nb_environment_reset
            self.previous_state = state
            action = self.poly_rl_alg.select_action(state, previous_action, tensor_board_writer=tensor_board_writer, step_number=step_number)
            action = torch.clamp(action, -1, 1).reshape(-1).numpy()
            return action
