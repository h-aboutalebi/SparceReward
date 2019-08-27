import argparse
from tensorboardX import SummaryWriter
import datetime
import time
import logging
import pickle

from engine.run_RL import Run_RL

logger = logging.getLogger(__name__)
import sys
import gym
import roboschool
import numpy as np
import torch
import random
import os

from engine.algorithms.param_noise import *
from engine.utils.setting_tools import get_agent_type
from engine.utils.replay_memory import ReplayMemory, Transition
from policy_engine.mod_reward import *
from policy_engine.poly_rl import *
from policy_engine.ddpg import DDPG
from policy_engine.div_ddpg_actor import *
from policy_engine.normalized_actions import NormalizedActions
from policy_engine.ounoise import OUNoise

parser = argparse.ArgumentParser(description='PyTorch poly Rl exploration implementation')

# *********************************** General Setting ********************************************

parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_exploration_policy',
                    help='output path for files produced by the agent')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')

# *********************************** Environment Setting ********************************************

parser.add_argument('--env_name', default="RoboschoolHalfCheetah-v1",
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
memory = ReplayMemory(args.replay_size)
# sets agent type:
agent = get_agent_type(args, env)
reward_modifier=Reward_Zero_Sparce(args.threshold_sparcity,args.sparse_reward)
new_run = Run_RL(num_steps=args.num_steps, update_interval=args.update_interval,eval_interval=args.eval_interval,
                 mini_batch_size=args.mini_batch_size, agent=agent, env=env,memory=memory)
Final_results = {"reward": [], "modified_reward": [], "poly_rl_ratio": {"ratio": [], "step_number": [], "epoch": []}}
start_time = time.time()
new_run.run(start_time,writer)
for i_episode in range(args.num_episodes):
    total_numsteps_episode = 0
    state = torch.Tensor([env.reset()])
    if (args.poly_rl_exploration_flag):
        poly_rl_alg.reset_parameters_in_beginning_of_episode(i_episode + 2)

    episode_reward = 0
    previous_action = None
    previous_state = state
    counter = 0
    while (counter < args.num_steps):
        total_numsteps += 1
        action = agent.select_action(state=state, action_noise=ounoise, previous_action=previous_action, tensor_board_writer=writer,
                                     step_number=total_numsteps, param_noise=param_noise)
        previous_action = action
        next_state, reward, done, info_ = env.step(action.cpu().numpy()[0])
        total_numsteps_episode += 1
        episode_reward += reward
        action = torch.Tensor(action.cpu())
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        modified_reward, flag_absorbing_state = make_reward_sparse(env=env, flag_sparse=args.sparse_reward, reward=reward,
                                                                   threshold_sparcity=args.threshold_sparcity,
                                                                   negative_reward_flag=args.reward_negative, num_steps=args.num_steps)
        modified_reward = torch.Tensor([modified_reward])
        memory.push(state, action, mask, next_state, modified_reward)
        previous_state = state
        state = next_state
        if (args.poly_rl_exploration_flag and poly_rl_alg.Update_variable):
            poly_rl_alg.update_parameters(previous_state=previous_state, new_state=state, tensor_board_writer=writer)

        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_parameters(batch, tensor_board_writer=writer,
                                                                  episode_number=i_episode)
                if args.param_noise and args.algo == "DDPG":
                    agent.perturb_actor_parameters(param_noise)
                updates += 1
        # if the environemnt should be reset, we break
        if done or flag_absorbing_state:
            break
        counter += 1

    writer.add_scalar('reward/train', episode_reward, i_episode)
    rewards.append(episode_reward)
    if args.param_noise:
        episode_transitions = memory.memory[memory.position - counter:memory.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(), unperturbed_actions.cpu().numpy())
        param_noise.adapt(ddpg_dist)

    if (args.poly_rl_exploration_flag):
        Final_results["poly_rl_ratio"]["ratio"].append(agent.number_of_time_target_policy_is_called)
        Final_results["poly_rl_ratio"]["step_number"].append(total_numsteps)
        Final_results["poly_rl_ratio"]["epoch"].append(i_episode)
    # print(Final_results)

    # This section is for computing the target policy perfomance
    # The environment is reset every 10 episodes automatically and we compute the target policy reward.
    if i_episode % 10 == 0:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        episode_modified_reward = 0
        counter = 0
        while (counter < args.num_steps):
            action = agent.select_action_from_target_actor(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_reward += reward
            modified_reward, flag_absorbing_state = make_reward_sparse(env=env, flag_sparse=args.sparse_reward, reward=reward,
                                                                       threshold_sparcity=args.threshold_sparcity,
                                                                       negative_reward_flag=args.reward_negative, num_steps=args.num_steps)

            episode_modified_reward += modified_reward
            next_state = torch.Tensor([next_state])
            state = next_state
            if done or flag_absorbing_state:
                break
            counter += 1

        writer.add_scalar('real_reward/test', episode_reward, i_episode)
        writer.add_scalar('reward_modified/test', episode_modified_reward, i_episode)
        time_len = time.time() - start_time
        start_time = time.time()
        rewards.append(episode_reward)

        Final_results["reward"].append(episode_reward)
        Final_results["modified_reward"].append(episode_modified_reward)
        last_x_body = env.env.body_xyz[0]
        writer.add_scalar('x_body', last_x_body, i_episode)
        logger.info(
            "Episode: {}, time:{}, numsteps in the episode: {}, total steps so far: {}, x_body: {}, reward: {}, modified_reward {}".format(
                i_episode, time_len, total_numsteps_episode, total_numsteps, last_x_body, episode_reward, episode_modified_reward))

    with open(file_path_results + '/result_reward0.pkl', 'wb') as handle:
        pickle.dump(Final_results, handle)
env.close()
