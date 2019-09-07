class AbstractAgent():

    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def select_action(self, state, tensor_board_writer=None, step_number=None):
        raise NotImplementedError

    def select_action_target(self, state, tensor_board_writer=None, step_number=None):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError