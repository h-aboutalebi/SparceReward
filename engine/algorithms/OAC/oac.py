from engine.algorithms.OAC.networks import FlattenMlp
from engine.algorithms.OAC.trainer.policies import TanhGaussianPolicy, MakeDeterministic
from engine.algorithms.OAC.trainer.trainer import SACTrainer


class OAC:

    def __init__(self, state_dim, action_dim, beta_UB, delta, oac_target_tau=5e-3, gamma=0.99,
                 oac_policy_lr=3E-4, target_update_period=1,oac_qf_lr=3E-4,reward_scale=1,
                 use_automatic_entropy_tuning=True,layer_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_size = layer_size
        self.q_producer = self.get_q_producer(state_dim, action_dim, hidden_sizes=[layer_size, layer_size])
        self.policy_producer = self.get_policy_producer(state_dim, action_dim, hidden_sizes=[layer_size, layer_size])
        self.trainer=SACTrainer(self.policy_producer, self.q_producer, action_dim, discount=gamma, soft_target_tau=oac_target_tau,
                                target_update_period=target_update_period, policy_lr=oac_policy_lr, qf_lr=oac_qf_lr,reward_scale=reward_scale,
                                use_automatic_entropy_tuning=use_automatic_entropy_tuning)
        self.algorithm=


    def get_q_producer(self, obs_dim, action_dim, hidden_sizes):
        def q_producer():
            return FlattenMlp(input_size=obs_dim + action_dim,
                              output_size=1,
                              hidden_sizes=hidden_sizes, )

        return q_producer

    def get_policy_producer(self, obs_dim, action_dim, hidden_sizes):
        def policy_producer(deterministic=False):
            policy = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
            )

            if deterministic:
                policy = MakeDeterministic(policy)

            return policy

        return policy_producer
