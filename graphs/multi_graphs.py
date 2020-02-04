from glob import glob
import math
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

# directory path that contains all the folder of pkl files
from graphs.func_graph import get_result_file, get_x, initilize_plt_conf, create_graph

directory_path = "/Users/hosseinaboutalebi/Desktop/myfigures"

#### performance graphs ####
result_folders = glob(directory_path + "/*/")

# initilizes the config of plt
colors = ['b', 'g', 'r', 'm', 'y', 'c']
smoothness=5
f = plt.figure(1)
create_graph(plt=plt, target="mod_reward", plt_figure=f, y_label="reward", x_label="step", result_folders=result_folders
             , colors=colors,smoothness=smoothness)
#### exploration graph in polyrl ####
g = plt.figure(2)
create_graph(plt=plt, target="poly_exploration", plt_figure=g, y_label="target policy percentage", x_label="step", result_folders=result_folders
             , colors=colors,smoothness=0, folder_name_cons="poly")

