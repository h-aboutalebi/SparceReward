from engine.algorithms.OAC.networks import FlattenMlp
from engine.algorithms.OAC.trainer.policies import TanhGaussianPolicy, MakeDeterministic


class OAC:

    def __init__(self, state_dim, action_dim, layer_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_size = layer_size
        self.q_producer = self.get_q_producer(state_dim, action_dim, hidden_sizes=[layer_size, layer_size])

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
