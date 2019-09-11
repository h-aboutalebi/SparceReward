import os
import shutil
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
import random


# implements the PolyRL algorithm
# It is implemented based on psudo code of algorithm
class PolyRL():

    def __init__(self, gamma,nb_actions,nb_observations,max_action,min_action, lambda_=0.08, betta=0.001, epsilon=0.999, sigma_squared=0.04,
                 actor_target_function=None):
        self.epsilon = epsilon
        self.number_of_time_PolyRL__update_parameter_is_called=0
        self.number_of_time_target_policy_is_called=0
        self.gamma = gamma
        self.lambda_ = lambda_
        self.sigma_squared = sigma_squared
        self.nb_actions = env.action_space.shape[0]
        self.nb_observations = self.env.observation_space.shape[0]
        self.max_action_limit = float(max(env.action_space.high))
        self.min_action_limit = float(min(env.action_space.low))
        self.betta = betta
        self.actor_target_function = actor_target_function
        self.number_of_goal = 0
        self.i = 1
        self.Update_variable=True
        self.g = 0
        self.C_vector = torch.zeros(1,self.nb_observations)
        self.delta_g = 0
        self.b = 0
        self.B_vector = torch.zeros(1, self.nb_observations)
        self.C_theta = 0.001
        self.L = -1
        self.U = 1
        self.t = 0  # to account for time step
        self.w_old = torch.zeros(1, self.nb_observations)
        self.w_new = torch.zeros(1, self.nb_observations)
        self.eta = None
        self.should_use_target_policy=False

    def select_action(self, state, previous_action,tensor_board_writer,step_number):
        self.Update_variable = True
        if (self.should_use_target_policy is True):
            k=random.uniform(0, 1)
            if(k<=self.epsilon):
                self.number_of_time_target_policy_is_called += 1
                tensor_board_writer.add_scalar('number_of_time_target_policy__exploration_is_called', self.number_of_time_target_policy_is_called,
                                               step_number)
                action = self.actor_target_function(state)
                return action
            else:
                self.should_use_target_policy = True
        elif (self.t == 0):
            return torch.Tensor(1, self.nb_actions).uniform_(self.min_action_limit, self.max_action_limit)

        elif (((self.delta_g < self.U) and (self.delta_g > self.L) and (self.C_theta>0)) or self.i==1):
            self.eta = abs(np.random.normal(self.lambda_, np.sqrt(self.sigma_squared)))
            return self.sample_action_algorithm(previous_action)

        else:
            self.number_of_time_target_policy_is_called += 1
            tensor_board_writer.add_scalar('number_of_time_target_policy__exploration_is_called', self.number_of_time_target_policy_is_called,step_number)
            action = self.actor_target_function(state)
            self.reset_parameters_PolyRL()
            self.Update_variable = False
            self.should_use_target_policy = True
            return action

    # This function resets parameters of PolyRl every episode. Should be called in the beggining of every episode
    def reset_parameters_in_beginning_of_episode(self,episode_number):
        self.should_use_target_policy=False
        self.epsilon = 1 - np.exp(-self.betta*episode_number)
        self.i = 1
        self.g = 0
        self.C_vector = torch.zeros(1, self.nb_observations)
        self.delta_g = 0
        self.b = 0
        self.B_vector = torch.zeros(1, self.nb_observations)
        self.C_theta = 0.001
        self.L = -1
        self.U = 1
        self.t = 0  # to account for time step
        self.w_old = torch.zeros(1, self.nb_observations)
        self.w_new = torch.zeros(1, self.nb_observations)
        self.eta = None

    def sample_action_algorithm(self, previous_action):
        previous_action=previous_action.cpu()
        P = torch.FloatTensor(previous_action.shape[1]).uniform_(float(self.min_action_limit), float(self.max_action_limit))
        D = torch.dot(P, previous_action.reshape(-1)).item()
        norm_previous_action = np.linalg.norm(previous_action.numpy(), ord=2)
        V_p = (D / norm_previous_action ** 2) * previous_action
        V_r = P - V_p
        l = np.linalg.norm(V_p.numpy(), ord=2) * np.tan(self.eta)
        k = l / np.linalg.norm(V_r.numpy(), ord=2)
        Q = k * V_r + V_p
        if (D > 0):
            action = Q
        else:
            action = -Q
        action = np.clip(action.numpy(), self.min_action_limit, self.max_action_limit)
        self.i += 1
        return torch.from_numpy(action).reshape(1, action.shape[1])

    def update_parameters(self, previous_state, new_state,tensor_board_writer=None):
        self.w_old = self.w_new
        norm_w_old = np.linalg.norm(self.w_old.numpy(), ord=2)
        self.w_new = new_state - previous_state
        norm_w_new = np.linalg.norm(self.w_new.numpy(), ord=2)
        self.B_vector = self.B_vector + self.i * self.w_new
        if (self.i != 1):
            Delta1 = previous_state - self.C_vector
            self.old_g = self.g
            self.g = ( (self.i-2)/(self.i-1)) * self.g + (1 / self.i) * np.linalg.norm(Delta1.numpy(), ord=2) ** 2
            self.delta_g=self.g-self.old_g
            old_C_theta=self.C_theta
            self.C_theta = ((self.i - 2) * self.C_theta + torch.dot(self.w_new.reshape(-1),
                                                                    self.w_old.reshape(-1)).item() / (
                                    norm_w_new * norm_w_old)) / (self.i - 1)
            K = 0
            for j in range(1, self.i):
                if(self.C_theta==1):
                    K=K+j
                elif(self.C_theta>0):
                    K = K + j * np.exp((j - self.i) / (1 / np.log(self.C_theta)))
            norm_B_vector = np.linalg.norm(self.B_vector.numpy(), ord=2)
            last_term = (1 / (self.i - 1)) * self.old_g

            # Upper bound and lower bound are computed here
            self.U = (1 / ((self.i ** 3) * (1 - self.epsilon))) * (
                    (self.i ** 2) * self.b + (norm_B_vector ** 2) + 2 * self.i * self.b * K) - last_term

            tensor_board_writer.add_scalar('Upper_bound_PolyRL', self.U, self.number_of_time_PolyRL__update_parameter_is_called)

            self.L = (1 - np.sqrt(2 * self.epsilon)) * (
                    self.b / self.i + (((self.i - 1) * (self.i - 2)) / self.i ** 2) * self.b * self.compute_correlation_decay() + (1 / self.i ** 3) * norm_B_vector ** 2) - last_term

            self.L = max(0, self.L)

            # logger.info("@Upper bound value: {}, @Lower bound value: {}".format(self.U,self.L))


            tensor_board_writer.add_scalar('Lower_bound_PolyRL', self.L, self.number_of_time_PolyRL__update_parameter_is_called)
            tensor_board_writer.add_scalar('Delta_bounds_PolyRL', self.U-self.L, self.number_of_time_PolyRL__update_parameter_is_called)


        self.number_of_time_PolyRL__update_parameter_is_called+=1
        self.b = ((self.i - 1) * self.b + norm_w_new ** 2) / self.i
        self.C_vector = ((self.i - 1) * self.C_vector + previous_state) / self.i
        self.t += 1

    def compute_correlation_decay(self):
        if(self.C_theta==1):
            return 1
        else:
            Lp=1 / np.log(self.C_theta)
            return np.exp((-abs(self.i - 1)) / Lp)


    # This function resets the parameters of class
    def reset_parameters_PolyRL(self):
        self.i = 1
        self.g = 0
        self.C_vector = torch.zeros(self.nb_observations)
        self.delta_g = 0
        self.b = 0
        self.B_vector = torch.zeros(self.nb_observations)
        self.C_theta = 0.001
        self.L = -1
        self.U = 1
        self.t = 0
        self.w_new = torch.zeros(self.nb_observations)
        self.w_old = torch.zeros(self.nb_observations)
