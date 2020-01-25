import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from engine.algorithms.DDPG.ddpg import DDPG
from engine.algorithms.abstract_agent import AbstractAgent


class DDPG_No_Noise(DDPG):
    def __init__(self, state_dim, action_dim, max_action, expl_noise, action_high, action_low, tau,device,lr_actor):
        super(DDPG_No_Noise, self).__init__(state_dim=state_dim, action_dim=action_dim,
                                    max_action=max_action,device=device,action_high=action_high,
                                    action_low=action_low,expl_noise=expl_noise,lr_actor=lr_actor,
                                    tau=tau)

    def select_action(self, state, tensor_board_writer=None, previous_action=None, step_number=None):
        state = np.array(state)
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


