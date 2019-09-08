import logging
import torch
import numpy as np
import time

from engine.utils.env_tools import get_current_pose

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
        self.nb_env_reset = 0
        self.reward_modifier = reward_modifier
        self.initial_x = None

    def run(self, start_time, writer):
        logger.info("Learning has started ...")
        total_reward = 0
        total_modified_reward = 0
        states = []
        actions = [None]
        env_is_reset = True
        for step_number in range(self.num_steps):
            if (env_is_reset is True):
                self.nb_env_reset += 1
                logger.debug("Environment has been reset (done is True). Counter = {}".format(self.nb_env_reset))
                self.initial_x = get_current_pose(self.env)
                states.append(self.env.reset())
                env_is_reset = False
            action = self.agent.select_action(state=states[-1], previous_action=actions[-1], tensor_board_writer=writer
                                              , step_number=step_number)
            next_state, reward, done, info_ = self.env.step(action)
            if (done):
                env_is_reset = True
            states.append(next_state)
            actions.append(action)
            modified_reward = self.reward_modifier.make_reward_sparse(reward, self.initial_x)
            self.memory.add((states[-2], states[-1], action, reward, done))
            total_reward += reward
            total_modified_reward += modified_reward
            self.update_agent(step_number, writer,env_is_reset)
            start_time = self.evaluate_policy(start_time, step_number, writer)
            if (step_number % 10 == 0):
                writer.add_scalar('raw_reward/train', total_reward, step_number)
                writer.add_scalar('mod_reward/train', total_modified_reward, step_number)

    def update_agent(self, step_number, writer,env_is_reset):
        if (step_number % self.update_interval == 0 and step_number > self.mini_batch_size):
            # logger.info("Target policy agent has been updated")
            self.agent.train(replay_buffer=self.memory, writer=writer,
                             step_number=step_number,batch_size=self.mini_batch_size,
                             env_reset=env_is_reset)

    # This function evaluates the target policy if the eval_interval has reached
    def evaluate_policy(self, start_time, step_number, writer):
        total_reward = 0
        total_modified_reward = 0
        done = False
        if self.timesteps_since_eval >= self.eval_interval:
            logger.info("Evaluating target policy ...")
            self.initial_x = get_current_pose(self.env)
            state = self.env.reset()
            actions = [None]
            self.timesteps_since_eval = 0
            while (True):
                action = self.agent.select_action_target(state=state, previous_action=actions[-1], tensor_board_writer=writer)
                state, reward, done, info_ = self.env.step(action)
                total_reward += reward
                actions.append(action)
                modified_reward = self.reward_modifier.make_reward_sparse(reward, self.initial_x)
                total_modified_reward += modified_reward
                if done:
                    break
            time_elapsed = time.time() - start_time
            logger.info("Elapsed time:{} | Number of steps: {} | Raw reward: {} | Modified reward: {}"
                        .format(time_elapsed, step_number, total_reward, total_modified_reward))
            writer.add_scalar('raw_reward/test', total_reward, step_number)
            writer.add_scalar('mod_reward/test', total_modified_reward, step_number)
            return time.time()
        self.timesteps_since_eval += 1
        return start_time
