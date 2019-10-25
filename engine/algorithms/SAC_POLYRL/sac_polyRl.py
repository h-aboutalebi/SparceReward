import torch

from engine.algorithms.SAC.sac import SAC


class SAC_Poly_RL(SAC):



    def select_action(self, state, tensor_board_writer=None, step_number=None):
        self.counter_actions += 1
        state = torch.Tensor(state).reshape(1, -1)
        action, _, _ = self.policy.sample(state)
        if (self.start_steps < self.counter_actions):
            return action.detach().cpu().numpy()[0]
        else:
            return self.action_space.sample()