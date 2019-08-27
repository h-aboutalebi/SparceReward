import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class Run_RL():

    def __init__(self, reward_modifier,num_steps, memory, update_interval, eval_interval, mini_batch_size, agent, env):
        self.num_steps = num_steps
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.agent = agent
        self.mini_batch_size = mini_batch_size
        self.env = env
        self.timesteps_since_eval = 0
        self.memory = memory
        self.reward_modifier=reward_modifier

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
                states = torch.Tensor([self.env.reset()])
            action = self.agent.select_action(state=states[-1], previous_action=actions[-1], tensor_board_writer=writer
                                              , step_number=step_number)
            next_state, reward, done, info_ = self.env.step(action.cpu().numpy()[0])
            states.append(torch.Tensor([next_state]))
            actions.append(torch.Tensor(action.cpu()))
            mask = torch.Tensor([not done])
            modified_reward = self.reward_modifier.make_reward_sparse(reward)
            self.save_in_memory(states, actions, torch.Tensor([modified_reward]), mask)
            episode_reward += reward
            self.update_agent(step_number)
            self.evaluate_policy()

    def save_in_memory(self, states, actions, reward, mask):
        self.memory.push(states[-2], actions[-1], mask, states[-1], reward)

    def update_agent(self, step_number):
        if (step_number % self.update_interval == 0 and step_number > self.mini_batch_size):
            logger.info("Target policy agent has been updated")

    # This function evaluates the target policy if the eval_interval has reached
    def evaluate_policy(self):
        if self.timesteps_since_eval >= self.eval_interval:
            self.timesteps_since_eval = 0
            logger.info("Target policy agent has been evaluated")
        self.timesteps_since_eval += 1
