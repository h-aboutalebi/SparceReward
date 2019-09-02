import logging
import torch
import numpy as np

from engine.utils.env_tools import get_current_pose
from engine.utils.replay_memory import ReplayMemory, Transition

logger = logging.getLogger(__name__)


class Run_RL():

    def __init__(self, reward_modifier, num_steps, memory, update_interval, eval_interval,
                 mini_batch_size, agent, env):
        self.num_steps = num_steps
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.agent = agent
        self.mini_batch_size = mini_batch_size
        self.env = env
        self.timesteps_since_eval = 0
        self.memory = memory
        self.reward_modifier = reward_modifier
        self.initial_x = None

    def run(self, start_time, writer):
        logger.info("Learning has started ...")
        rewards = np.array([])
        episode_reward = 0
        updates = 0
        states = [torch.Tensor([self.env.reset()])]
        actions = [torch.Tensor([0.0 for _ in range(self.env.shape[0])])]
        env_is_reset = True
        for step_number in range(self.num_steps):
            if (env_is_reset is True):
                logger.info("Environment has been reset (done is True)")
                self.initial_x = get_current_pose(self.env)
                states.append(torch.Tensor([self.env.reset()]))
                env_is_reset = False
            action = self.agent.select_action(state=states[-1], previous_action=actions[-1], tensor_board_writer=writer
                                              , step_number=step_number)
            next_state, reward, done, info_ = self.env.step(action.cpu().numpy()[0])
            states.append(torch.Tensor([next_state]))
            actions.append(torch.Tensor(action.cpu()))
            mask = torch.Tensor([not done])
            modified_reward = self.reward_modifier.make_reward_sparse(reward, self.initial_x)
            self.save_in_memory(states, actions, torch.Tensor([modified_reward]), mask)
            episode_reward += reward
            self.update_agent(step_number, writer)
            self.evaluate_policy(states)

    def save_in_memory(self, states, actions, reward, mask):
        self.memory.push(states[-2], actions[-1], mask, states[-1], reward)

    def update_agent(self, step_number, writer):
        if (step_number % self.update_interval == 0 and step_number > self.mini_batch_size):
            logger.info("Target policy agent has been updated")
            transitions = self.memory.sample(self.mini_batch_size)
            batch = Transition(*zip(*transitions))
            value_loss, policy_loss = self.agent.update_parameters(batch, tensor_board_writer=writer,
                                                                   episode_number=step_number)

    # This function evaluates the target policy if the eval_interval has reached
    def evaluate_policy(self,states):
        Total_reward = 0
        Total_modified_reward = 0
        done = False
        if self.timesteps_since_eval >= self.eval_interval:
            logger.info("Evaluating target policy ...")
            self.initial_x = get_current_pose(self.env)
            states.append(torch.Tensor([self.env.reset()]))
            self.timesteps_since_eval = 0
            while (True):
                action = self.agent.select_action_from_target_actor(state)
                next_state, reward, done, _ = self.env.step(action.cpu().numpy()[0])
                Total_reward += reward
                modified_reward = self.reward_modifier.make_reward_sparse(reward, self.initial_x)
                Total_modified_reward += modified_reward
                next_state = torch.Tensor([next_state])
                state = next_state
                if done:
                    return
        self.timesteps_since_eval += 1
