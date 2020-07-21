import argparse
from tensorboardX import SummaryWriter
import datetime
import time
import logging

from engine.reward_modifier.reward_zero_sparce import Reward_Zero_Sparce
from engine.run_RL import Run_RL
from graphs.singly_graph import Create_Graph

logger = logging.getLogger(__name__)
import gym
import numpy as np
import torch
import random
import os

from engine.utils.setting_tools import get_agent_type

parser = argparse.ArgumentParser(description='PyTorch poly Rl exploration implementation')

# *********************************** General Setting ********************************************

parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_exploration_policy',
                    help='output path for files produced by the agent')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--del_tensor_file', action='store_false',
                    help='whether to delete the tensorboard file after run completes')

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

parser.add_argument('--sparse_reward', action='store_true',
                    help='for making reward sparse. Default=False')

parser.add_argument('--threshold_sparcity', type=float, default=1.15, metavar='G',
                    help='threshold_sparcity for rewards (default: 0.15)')

# *********************************** Algorithm Setting ********************************************

parser.add_argument('--algo', default='DDPG',
                    help='Current supported algorithms to use: DDPG | DDPG_DIV | DDPG_PARAM | '
                         'DDPG_POLYRL | DDPG_NO_NOISE | DDPG_OU_NOISE | SAC | SAC_POLYRL')

# *********************************** DDPG Setting ********************************************

# This is the factor for updating the target policy with delay based on behavioural policy
parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise

parser.add_argument("--lr_actor", default=1e-4, type=float)  # Std of Gaussian exploration noise

parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate)

parser.add_argument('--buffer_size', type=int, default=1e6)

# Important: batch size here is different in semantics
parser.add_argument('--mini_batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 1.0e9). This is different from usual deep learning work'
                         'where batch size infers parallel processing. Here, we currently do not have that as'
                         'we update our parameters sequentially. Here, batch_size means minimum length that '
                         'memory replay should have before strating to update model parameters')

# *********************************** DDPG OU NOISE Setting ********************************************

parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')

# *********************************** Param Noise DDPG Setting ********************************************

# Note: The following noise are for the behavioural policy of the pure DDPG without the poly_rl policy

parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')

parser.add_argument('--initial_stdev', type=float, default=1e-4)

# *********************************** DIV DDPG Setting ********************************************

parser.add_argument('--linear_flag_div', action='store_true')

parser.add_argument('--phi_div', type=float, default=0.5)

# *********************************** Poly_Rl Setting ********************************************

parser.add_argument('--betta', type=float, default=0.0001)

parser.add_argument('--epsilon', type=float, default=0)

parser.add_argument('--sigma_squared', type=float, default=0.00007)

parser.add_argument('--lambda_', type=float, default=0.035)

# *********************************** SAC Setting ********************************************

parser.add_argument("--gamma_sac", default=0.99, type=float)  # Std of Gaussian exploration noise

parser.add_argument("--tau_sac", default=0.005, type=float)  # Target network update rate

parser.add_argument("--alpha_sac", default=0.2, type=float)  # Target network update rate

parser.add_argument('--policy_sac', type=str, default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')

parser.add_argument('--start_steps', type=int, default=10000,
                    help='algorithm to use: Gaussian | Deterministic')

parser.add_argument('--automatic_entropy_tuning', action='store_true',
                    help='for making reward sparse. Default=False')

# *********************************** SAC_Poly_Rl Setting ********************************************
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
logging.basicConfig(level=logging.DEBUG, filename=file_path_results + "/log.txt")
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
if (args.del_tensor_file):
    writer.STOP = True
    logger.info("Tensorboard is disabled!")

# *********************************** Environment Building ********************************************
env = gym.make(args.env_name)
env.seed(args.seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
logger.info(
    "action dimension: {} | State dimension: {} | Max action number: {}".format(action_dim, state_dim, max_action))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("device is set for: {}".format(device))

logger.info("Creating Agent ...")
# sets agent type:
agent, memory = get_agent_type(state_dim, action_dim, max_action, args, env, device)
reward_modifier = Reward_Zero_Sparce(env, args.threshold_sparcity, args.sparse_reward)
path_file_result = file_path_results + "/results.pkl"
new_run = Run_RL(reward_modifier=reward_modifier, num_steps=int(args.num_steps), update_interval=args.update_interval,
                 eval_interval=args.eval_interval,
                 mini_batch_size=args.mini_batch_size, agent=agent, env=env, memory=memory,
                 path_file_result=path_file_result)
start_time = time.time()
new_run.run(start_time, writer)
logger.info("results saved in file {}".format(path_file_result))
Create_Graph(path_pkl=path_file_result, path_image=file_path_results, name=args.algo)
