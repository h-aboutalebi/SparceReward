import logging
import torch
import numpy as np
import time

from engine.utils.setting_tools import select_action_agent, post_update_agent, select_action_target

try:
    import cPickle as pk
except:
    import pickle as pk

from engine.utils.env_tools import get_current_pose

logger = logging.getLogger(__name__)


class Run_RL():

    def __init__(self, reward_modifier, num_steps, memory, update_interval, eval_interval,
                 mini_batch_size, agent, env, path_file_result):
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
        self.result = []
        self.path_file_result = path_file_result

    def run(self, start_time, writer):
        logger.info("Learning has started ...")
        total_reward = 0
        total_modified_reward = 0
        states = []
        actions = [None]
        env_is_reset = True
        previous_total_reward = 0
        previous_total_modified_reward = 0
        for step_number in range(self.num_steps):
            if (env_is_reset is True):
                self.nb_env_reset += 1
                logger.debug(
                    "Environment has been reset (done is True). Counter = {} | Num_steps = {} | episode_tot_reward = {} | episode_tot_mod_reward = {}".format(
                        self.nb_env_reset, step_number, total_reward - previous_total_reward, total_modified_reward - previous_total_modified_reward))
                previous_total_reward = total_reward
                previous_total_modified_reward = total_modified_reward
                states.append(self.env.reset())
                self.initial_x = get_current_pose(self.env)
                env_is_reset = False
            # update_parameters()
            action = select_action_agent(state=states[-1], previous_action=actions[-1], tensor_board_writer=writer
                                         , step_number=step_number, nb_environment_reset=self.nb_env_reset, agent=self.agent)
            next_state, reward, done, info_ = self.env.step(action)

            post_update_agent(agent=self.agent, previous_state=states[-1], next_state=next_state,
                              done=done, step_number=step_number, writer=writer)
            if (done):
                env_is_reset = True
            states.append(next_state)
            actions.append(action)
            modified_reward = self.reward_modifier.make_reward_sparse(reward, self.initial_x)
            self.memory.add((states[-2], states[-1], action, modified_reward, done))
            total_reward += reward
            total_modified_reward += modified_reward
            self.update_agent(step_number, writer, env_is_reset)
            start_time = self.evaluate_policy(start_time, step_number, writer)
            if (step_number % 10 == 0 and writer.STOP == False):
                writer.add_scalar('raw_reward/train', total_reward, step_number)
                writer.add_scalar('mod_reward/train', total_modified_reward, step_number)
        self.save_results()

    def save_results(self):
        with open(self.path_file_result, "wb") as input_file:
            pk.dump(self.result, input_file)

    def update_agent(self, step_number, writer, env_is_reset):
        if (step_number % self.update_interval == 0 and step_number > self.mini_batch_size):
            # logger.info("Target policy agent has been updated")
            self.agent.train(replay_buffer=self.memory, writer=writer,
                             step_number=step_number, batch_size=self.mini_batch_size,
                             env_reset=env_is_reset)

    # This function evaluates the target policy if the eval_interval has reached
    def evaluate_policy(self, start_time, step_number, writer):
        total_reward = 0
        total_modified_reward = 0
        done = False
        if self.timesteps_since_eval >= self.eval_interval:
            logger.info("Evaluating target policy ...")
            state = self.env.reset()
            self.initial_x = get_current_pose(self.env)
            actions = [None]
            self.timesteps_since_eval = 0
            while (True):
                action = select_action_target(state=state, previous_action=actions[-1], tensor_board_writer=writer
                                              , step_number=step_number, nb_environment_reset=self.nb_env_reset, agent=self.agent)
                state, reward, done, info_ = self.env.step(action)
                total_reward += reward
                actions.append(action)
                modified_reward = self.reward_modifier.make_reward_sparse(reward, self.initial_x)
                total_modified_reward += modified_reward
                if done:
                    break
            time_elapsed = time.time() - start_time
            percentage_poly_exploration = getattr(self.agent, "get_exploration_percentage", None)
            if (percentage_poly_exploration is not None):
                self.result.append({"step_nb": step_number, "raw_reward": total_reward, "mod_reward": total_modified_reward, "poly_exploration": percentage_poly_exploration()})
            else:
                self.result.append({"step_nb": step_number, "raw_reward": total_reward, "mod_reward": total_modified_reward})
            logger.info("Elapsed time:{} | Number of steps: {} | Raw reward: {} | Modified reward: {}"
                        .format(time_elapsed, step_number, total_reward, total_modified_reward))
            if (writer.STOP == False):
                writer.add_scalar('raw_reward/test', total_reward, step_number)
                writer.add_scalar('mod_reward/test', total_modified_reward, step_number)
            self.save_results()
            return time.time()
        self.timesteps_since_eval += 1
        return start_time
