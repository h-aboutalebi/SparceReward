

def get_current_pose(env):
    try:
        return env.env.sim.data.qpos[0]
    except:
        print("sim is not supported")
        return 0
