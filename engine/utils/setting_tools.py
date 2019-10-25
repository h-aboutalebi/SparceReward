import logging

from engine.algorithms.DDPG.ddpg import DDPG
from engine.algorithms.DDPG.replay_memory import ReplayBuffer
from engine.algorithms.DDPG_PARAMNOISE.ddpg_param_noise import DDPG_Param_Noise
from engine.algorithms.DDPG_POLYRL.ddpg import DDPGPolyRL
from engine.algorithms.SAC.replay_momory import ReplayBuffer_SAC
from engine.algorithms.SAC.sac import SAC

logger = logging.getLogger(__name__)


def get_agent_type(state_dim, action_dim, max_action, args, env, device):
    agent = None
    # Adds noise to the selected action by the policy"
    if (args.algo == "DDPG"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPG(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                     action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                     device=device)
    elif (args.algo == "DDPG_PARAM"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPG_Param_Noise(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                                 action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                                 initial_stdev=args.initial_stdev, noise_scale=args.noise_scale, memory=memory,
                                 device=device)
    elif (args.algo == "DDPG_POLYRL"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPGPolyRL(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                           action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                           device=device, gamma=args.gamma, betta=args.betta, epsilon=args.epsilon,
                           sigma_squared=args.sigma_squared, lambda_=args.lambda_, nb_actions=env.action_space.shape[0],
                           nb_observations=env.observation_space.shape[0], min_action=float(min(env.action_space.low)))
    elif (args.algo == "SAC"):
        agent = SAC(state_dim, action_dim, max_action, action_space=env.action_space,
                    gamma=args.gamma_sac, alpha=args.alpha_sac, tau=args.tau_sac, policy=args.policy_sac, device=device)
        memory = ReplayBuffer_SAC(args.buffer_size)
    else:
        logger.info("Algorithm {} has not yet implemented! please select among 'DDPG, PARAM_DDPG, POLYRL_DDPG, DIV_DDPG'".format(args.algo))
        raise NotImplementedError
    logger.info("Algorithm {} has been initialized".format(args.algo))
    return agent, memory


def select_action_agent(state, previous_action, tensor_board_writer
                                              , step_number,nb_environment_reset,agent):
    if(type(agent).__name__=="DDPGPolyRL"):
        return agent.select_action(state, tensor_board_writer=tensor_board_writer, previous_action=previous_action,
                                   step_number=step_number,nb_environment_reset=nb_environment_reset)
    else:
        return agent.select_action(state, tensor_board_writer=tensor_board_writer, step_number=step_number)

def select_action_target(state, previous_action, tensor_board_writer
                                              , step_number,nb_environment_reset,agent):
    if(type(agent).__name__=="DDPGPolyRL"):
        return agent.select_action_target(state, tensor_board_writer=tensor_board_writer, previous_action=previous_action,
                                   step_number=step_number)
    else:
        return agent.select_action_target(state, tensor_board_writer=tensor_board_writer, step_number=step_number)


def post_update_agent(agent,previous_state,next_state,writer):
    if (type(agent).__name__ == "DDPGPolyRL"):
        agent.poly_rl_alg.update_parameters(previous_state=previous_state, new_state=next_state,
                                                 tensor_board_writer=writer)


