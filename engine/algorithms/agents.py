
class agents():

    def __init__(self,gamma,action_space,num_inputs):
        self.gamma=gamma
        self.action=action_space
        self.num_inputs=num_inputs

    def select_action(self,**kwargs):
        raise NotImplementedError

    def select_action_target(self,**kwargs):
        raise NotImplementedError