import logging

from engine.algorithms.DDPG.ddpg import DDPG
from engine.algorithms.DDPG.replay_memory import ReplayBuffer
from engine.algorithms.DDPG_DIV.ddpg_div import DivDDPGActor
from engine.algorithms.DDPG_No_Noise.ddpg_no_noise import DDPG_No_Noise
from engine.algorithms.DDPG_OU_NOISE.ddpg_ou_noise import DDPG_Ou_Noise
from engine.algorithms.DDPG_PARAMNOISE.ddpg_param_noise import DDPG_Param_Noise
from engine.algorithms.DDPG_POLYRL.ddpg import DDPGPolyRL
from engine.algorithms.SAC.replay_momory import ReplayBuffer_SAC
from engine.algorithms.SAC.sac import SAC
from engine.algorithms.SAC_POLYRL.sac_polyRl import SAC_Poly_RL

logger = logging.getLogger(__name__)


def get_agent_type(state_dim, action_dim, max_action, args, env, device):
    agent = None
    # Adds noise to the selected action by the policy"
    if (args.algo == "DDPG"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPG(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                     action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                     device=device, lr_actor=args.lr_actor)
    elif (args.algo == "DDPG_NO_NOISE"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPG_No_Noise(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                     action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                     device=device, lr_actor=args.lr_actor)
    elif(args.algo=="DDPG_OU_NOISE"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPG_Ou_Noise(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                              action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                              device=device, lr_actor=args.lr_actor,noise_scale=args.noise_scale,
                              num_steps=args.num_steps,final_noise_scale=args.final_noise_scale)
    elif (args.algo == "DDPG_DIV"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DivDDPGActor(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                     action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                     device=device, lr_actor=args.lr_actor, linear_flag=args.linear_flag_div,phi=args.phi_div)
    elif (args.algo == "DDPG_PARAM"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPG_Param_Noise(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                                 action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                                 initial_stdev=args.initial_stdev, noise_scale=args.noise_scale, memory=memory,
                                 device=device, lr_actor=args.lr_actor)
    elif (args.algo == "DDPG_POLYRL"):
        memory = ReplayBuffer(args.buffer_size)
        agent = DDPGPolyRL(state_dim, action_dim, max_action, expl_noise=args.expl_noise,
                           action_high=env.action_space.high, action_low=env.action_space.low, tau=args.tau,
                           device=device, gamma=args.gamma, betta=args.betta, epsilon=args.epsilon,
                           sigma_squared=args.sigma_squared, lambda_=args.lambda_, nb_actions=env.action_space.shape[0],
                           nb_observations=env.observation_space.shape[0], min_action=float(min(env.action_space.low)), lr_actor=args.lr_actor)
    elif (args.algo == "SAC"):
        agent = SAC(state_dim, action_dim, max_action, action_space=env.action_space,
                    gamma=args.gamma_sac, alpha=args.alpha_sac, tau=args.tau_sac, policy=args.policy_sac, device=device, start_steps=args.start_steps)
        memory = ReplayBuffer_SAC(args.buffer_size)
    elif (args.algo == "SAC_POLYRL"):
        agent = SAC_Poly_RL(state_dim, action_dim, max_action, action_space=env.action_space,
                    gamma=args.gamma_sac, alpha=args.alpha_sac, tau=args.tau_sac, policy=args.policy_sac, device=device, start_steps=args.start_steps,
                    betta = args.betta, epsilon = args.epsilon,sigma_squared = args.sigma_squared, lambda_ = args.lambda_, nb_actions = env.action_space.shape[0],
                    nb_observations =env.observation_space.shape[0], min_action = float(min(env.action_space.low)))
        memory = ReplayBuffer_SAC(args.buffer_size)
    else:
        logger.info("Algorithm {} has not yet implemented! please select among 'DDPG, PARAM_DDPG, POLYRL_DDPG, DIV_DDPG'".format(args.algo))
        raise NotImplementedError
    logger.info("Algorithm {} has been initialized".format(args.algo))
    return agent, memory


def select_action_agent(state, previous_action, tensor_board_writer
                        , step_number, nb_environment_reset, agent):
    if (type(agent).__name__ == "DDPGPolyRL"):
        return agent.select_action(state, tensor_board_writer=tensor_board_writer, previous_action=previous_action,
                                   step_number=step_number, nb_environment_reset=nb_environment_reset)
    elif(type(agent).__name__ == "SAC_Poly_RL"):
        return agent.mod_select_action(state, tensor_board_writer=tensor_board_writer, previous_action=previous_action,
                                   step_number=step_number, nb_environment_reset=nb_environment_reset)
    else:
        return agent.select_action(state, tensor_board_writer=tensor_board_writer, step_number=step_number)


def select_action_target(state, previous_action, tensor_board_writer
                         , step_number, nb_environment_reset, agent):
    if (type(agent).__name__ == "DDPGPolyRL"):
        return agent.select_action_target(state, tensor_board_writer=tensor_board_writer, previous_action=previous_action,
                                          step_number=step_number)
    else:
        return agent.select_action_target(state, tensor_board_writer=tensor_board_writer, step_number=step_number)


def post_update_agent(agent, previous_state, next_state, done,step_number,writer):
    if (type(agent).__name__ == "DDPGPolyRL" or type(agent).__name__ == "SAC_Poly_RL"):
        agent.poly_rl_alg.update_parameters(previous_state=previous_state, new_state=next_state,
                                            tensor_board_writer=writer)
    elif(type(agent).__name__ == "DDPG_Ou_Noise"):
        if(done):
            agent.update_action_noise(step_number)