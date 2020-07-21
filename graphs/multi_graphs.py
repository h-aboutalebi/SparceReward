
from glob import glob
import math
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

# directory path that contains all the folder of pkl files
from graphs.func_graph import get_result_file, get_x, initilize_plt_conf, create_graph

directory_path = "/Users/susanamin/PolyRL Project Figures/Ant_SAC"

#### performance graphs ####
result_folders = glob(directory_path + "/*/")

# initilizes the config of plt
colors = ['tab:orange', '#21618C', 'r', 'm', 'c', 'y']
smoothness=2
f = plt.figure(1)
create_graph(plt=plt, target="mod_reward", plt_figure= f, y_label="Reward", x_label="Number of Steps", result_folders=result_folders
             , colors=colors,smoothness=smoothness, directory_path = directory_path, name = 'Performance')
plt.show()
#### exploration graph in polyrl ####
g = plt.figure(2)
create_graph(plt=plt, target="poly_exploration", plt_figure= g, y_label="target policy percentage", x_label="Number of Steps", result_folders=result_folders
             , colors=colors,smoothness=0, folder_name_cons="poly", directory_path = directory_path, name = 'Percentage')


