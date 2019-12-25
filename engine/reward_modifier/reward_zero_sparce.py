from engine.utils.env_tools import get_current_pose


class Reward_Zero_Sparce():

    def __init__(self, env, sparcity_thereshold, sparcity_flag):
        self.sparcity_thereshold = sparcity_thereshold
        self.sparcity_flag = sparcity_flag
        self.env = env

    def make_reward_sparse(self, reward, initial_x_pose):
        if (self.sparcity_flag is False):
            return reward
        current_x_pose = get_current_pose(self.env)
        if (current_x_pose - initial_x_pose >= self.sparcity_thereshold):
            return 1
        else:
            return 0
