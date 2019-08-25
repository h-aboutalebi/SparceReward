import logging
import torch

logger = logging.getLogger(__name__)


class Run_RL():

    def __init__(self, num_steps, update_interval, mini_batch_size, agent,env):
        self.num_steps =num_steps
        self.update_interval =update_interval
        self.agent =agent
        self.batch_size=mini_batch_size
        self.env=env

    def run(self,start_time):
        logger.info("Learning has started ...")
        rewards = []
        updates = 0
        state = torch.Tensor([self.env.reset()])
        env_is_reset=True
        for step_number in range(self.num_steps):
            if(step_number %self.update_interval==0 and step_number>self.batch_size):
                self.update_agent()

