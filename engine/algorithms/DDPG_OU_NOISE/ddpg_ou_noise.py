import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from engine.algorithms.DDPG.ddpg import DDPG
from engine.algorithms.abstract_agent import AbstractAgent


class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

class DDPG_Ou_Noise(DDPG):

    def __init__(self, state_dim, action_dim, max_action, expl_noise, action_high, action_low, tau,device,lr_actor,noise_scale,num_steps,final_noise_scale):
        super(DDPG_Ou_Noise, self).__init__(state_dim=state_dim, action_dim=action_dim,
                                    max_action=max_action,device=device,action_high=action_high,
                                    action_low=action_low,expl_noise=expl_noise,lr_actor=lr_actor,
                                    tau=tau)
        self.noise_scale=noise_scale
        self.exploration_end=int(num_steps/10)
        self.final_noise_scale=final_noise_scale
        self.action_noise=OUNoise(action_dim)
        self.action_noise.scale = 0

    def select_action(self, state, tensor_board_writer=None, previous_action=None, step_number=None):
        state = np.array(state)
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action+ self.action_noise.noise()

    def update_action_noise(self,step_num):
        self.action_noise.scale = (self.noise_scale - self.final_noise_scale) * max(0, self.exploration_end -
                                                                      step_num) / self.exploration_end + self.final_noise_scale
        self.action_noise.reset()



