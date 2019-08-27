class Reward_Zero_Sparce():

    def __init__(self, sparcity_thereshold, sparcity_flag):
        self.sparcity_thereshold = sparcity_thereshold
        self.sparcity_flag = sparcity_flag

    def make_reward_sparse(self, reward):
        if (self.sparcity_flag is False):
            return reward
        if (reward > self.sparcity_thereshold):
            reward = reward
        else:
            reward = 0
