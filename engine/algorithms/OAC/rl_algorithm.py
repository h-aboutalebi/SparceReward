import abc
from collections import OrderedDict

from torch import nn as nn
from engine.algorithms.OAC.utils.logging import logger
import engine.algorithms.OAC.utils.eval_util as eval_util
from engine.algorithms.OAC.utils.rng import get_global_pkg_rng_state
import engine.algorithms.OAC.utils.pytorch_util as ptu

from tqdm import trange

import torch
import numpy as np
import random


class BatchRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            optimistic_exp_hp=None,
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.optimistic_exp_hp = optimistic_exp_hp

        """
        The class mutable state
        """
        self._start_epoch = 0

        """
        This class sets up the main training loop, so it needs reference to other
        high level objects in the algorithm
        But these high level object maintains their own states
        and has their own responsibilities in saving and restoring their state for checkpointing
        """
        self.trainer = trainer
        self.replay_buffer = replay_buffer

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):

        # Fill the replay buffer to a minimum before training starts
        if self.min_num_steps_before_training > self.replay_buffer.num_steps_can_sample():
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.trainer.policy,
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            # To evaluate the policy remotely,
            # we're shipping the policy params to the remote evaluator
            # This can be made more efficient
            # But this is currently extremely cheap due to small network size
            pol_state_dict = ptu.state_dict_cpu(self.trainer.policy)

            remote_eval_obj_id = self.remote_eval_data_collector.async_collect_new_paths.remote(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,

                deterministic_pol=True,
                pol_state_dict=pol_state_dict)

            gt.stamp('remote evaluation submit')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.trainer.policy,
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,

                    optimistic_exploration=self.optimistic_exp_hp['should_use'],
                    optimistic_exploration_kwargs=dict(
                        policy=self.trainer.policy,
                        qfs=[self.trainer.qf1, self.trainer.qf2],
                        hyper_params=self.optimistic_exp_hp
                    )
                )

                self.replay_buffer.add_paths(new_expl_paths)

                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)

            # Wait for eval to finish


            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        ray.get([self.remote_eval_data_collector.end_epoch.remote(epoch)])

        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        # We can only save the state of the program
        # after we call end epoch on all objects with internal state.
        # This is so that restoring from the saved state will
        # lead to identical result as if the program was left running.
        if epoch > 0:
            snapshot = self._get_snapshot(epoch)
            logger.save_itr_params(epoch, snapshot)
            gt.stamp('saving')

        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)

        write_header = True if epoch == 0 else False
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)



    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
