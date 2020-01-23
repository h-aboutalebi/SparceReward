class AbstractAgent():

    def __init__(self, state_dim, action_dim, max_action,device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device=device

    def select_action(self, *args,**kwargs):
        raise NotImplementedError

    def select_action_target(self, state, *args,**kwargs):
        raise NotImplementedError

    def train(self,  **kwargs):
        raise NotImplementedError
