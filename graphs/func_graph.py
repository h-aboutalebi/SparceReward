import pickle
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

def get_x(path_file):
    with open(path_file, 'rb') as f:
        example_dict = pickle.load(f)
    x = [x["step_nb"] for x in example_dict]
    return x

def get_result_file(results,path_file):
    with open(path_file, 'rb') as f:
        example_dict = pickle.load(f)
    mod_reward=[x["mod_reward"] for x in example_dict]
    y=make_smooth_line(mod_reward)
    results.append(y)
    return max_step_nb(example_dict),len(mod_reward)

def make_smooth_line(list):
    ysmoothed = gaussian_filter1d(list, sigma=1)
    return ysmoothed

def max_step_nb(example_dict):
    return (example_dict[-1]['step_nb']-example_dict[-2]['step_nb']-1)*(len(example_dict)+1)

def initilize_plt_conf(plt):
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
