import argparse
from tensorboardX import SummaryWriter
import datetime
import time
import logging
import pickle

from engine.algorithms.DDPG.replay_memory import ReplayBuffer
from engine.reward_modifier.reward_zero_sparce import Reward_Zero_Sparce
from engine.run_RL import Run_RL

logger = logging.getLogger(__name__)
import sys
import gym
import numpy as np
import torch
import random
import os

from engine.algorithms.param_noise import *
from engine.utils.setting_tools import get_agent_type

parser = argparse.ArgumentParser(description='PyTorch poly Rl exploration implementation')

# *********************************** General Setting ********************************************

parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_exploration_policy',
                    help='output path for files produced by the agent')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')

# *********************************** Environment Setting ********************************************

parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='name of the environment to run')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--num_steps', type=int, default=1.0e6, metavar='N',
                    help='max number of exploration steps (default: 1.0e6)')

parser.add_argument('--update_interval', type=int, default=1, metavar='N',
                    help='how often the target policy is updated (default: 1)')

parser.add_argument('--eval_interval', type=int, default=5.0e3, metavar='N',
                    help='how often the target policy is evaluated (default: 5.0e3)')

# *********************************** Reward Sparcity Setting ********************************************

parser.add_argument('--sparse_reward', action='store_false',
                    help='for making reward sparse. Default=True')

parser.add_argument('--threshold_sparcity', type=float, default=1.15, metavar='G',
                    help='threshold_sparcity for rewards (default: 0.15)')

# *********************************** Algorithm Setting ********************************************

parser.add_argument('--algo', default='DDPG',
                    help='algorithm to use: DDPG | PARAM_DDPG | DIV_DDPG | POLYRL_DDPG')

# *********************************** DDPG Setting ********************************************

# This is the factor for updating the target policy with delay based on behavioural policy
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')

parser.add_argument('--lr_actor', type=float, default=1e-4,
                    help='learning rate for actor policy')

parser.add_argument('--lr_critic', type=float, default=1e-4,
                    help='learning rate for critic policy')

parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')

# very important factor. Should be investigated in the future.
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')

# Important: batch size here is different in semantics
parser.add_argument('--mini_batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 1.0e9). This is different from usual deep learning work'
                         'where batch size infers parallel processing. Here, we currently do not have that as'
                         'we update our parameters sequentially. Here, batch_size means minimum length that '
                         'memory replay should have before strating to update model parameters')

# *********************************** Param Noise DDPG Setting ********************************************

# Note: The following noise are for the behavioural policy of the pure DDPG without the poly_rl policy
parser.add_argument('--ou_noise', type=bool, default=True,
                    help="This is the noise used for the pure version DDPG (without poly_rl_exploration)"
                         " where the behavioural policy has perturbation in only mean of target policy")

parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')

parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')

parser.add_argument('--param_noise_initial_stdev', type=float, default=1e-4)

# *********************************** DIV DDPG Setting ********************************************

parser.add_argument('--diverse_noise', action='store_true')

parser.add_argument('--linear_diverse_noise', action='store_true')

parser.add_argument('--phi', type=float, default=0.5)

parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')

# *********************************** Poly_Rl Setting ********************************************

parser.add_argument('--betta', type=float, default=0.0001)

parser.add_argument('--epsilon', type=float, default=0.999)

parser.add_argument('--sigma_squared', type=float, default=0.00007)

parser.add_argument('--lambda_', type=float, default=0.035)

args = parser.parse_args()

# sets the seed for making it comparable with other implementations
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# *********************************** Logging Config ********************************************
file_path_results = args.output_path + "/" + str(datetime.datetime.now()).replace(" ", "_")
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
os.mkdir(file_path_results)
logging.basicConfig(level=logging.INFO, filename=file_path_results + "/log.txt")
logging.getLogger().addHandler(logging.StreamHandler())

header = "===================== Experiment configuration ========================"
logger.info(header)
args_keys = list(vars(args).keys())
args_keys.sort()
max_k = len(max(args_keys, key=lambda x: len(x)))
for k in args_keys:
    s = k + '.' * (max_k - len(k)) + ': %s' % repr(getattr(args, k))
    logger.info(s + ' ' * max((len(header) - len(s), 0)))
logger.info("=" * len(header))

# for tensorboard
try:
    writer = SummaryWriter(logdir=file_path_results)
except:
    writer = SummaryWriter(file_path_results)

# *********************************** Environment Building ********************************************
env = gym.make(args.env_name)
env.seed(args.seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
logger.info(
    "action dimension: {} | State dimension: {} | Max action number: {}".format(action_dim, state_dim, max_action))

logger.info("Creating Agent ...")
memory = ReplayBuffer(args.buffer_size)
# sets agent type:
agent = get_agent_type(state_dim, action_dim, max_action, args, env)
reward_modifier = Reward_Zero_Sparce(env, args.threshold_sparcity, args.sparse_reward)
new_run = Run_RL(reward_modifier=reward_modifier, num_steps=args.num_steps, update_interval=args.update_interval, eval_interval=args.eval_interval,
                 mini_batch_size=args.mini_batch_size, agent=agent, env=env, memory=memory)
start_time = time.time()
new_run.run(start_time, writer)
