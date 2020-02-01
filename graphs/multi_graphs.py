from glob import glob
import math
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

# directory path that contains all the folder of pkl files
from graphs.func_graph import get_result_file, get_x, initilize_plt_conf

directory_path = "/Users/hosseinaboutalebi/Desktop/myfigures"
colors = ['b', 'g', 'r', 'm', 'y', 'c']

result_folders = glob(directory_path + "/*/")
dict_results = {}
line_names = []
min_x_axis = math.inf

# initilizes the config of plt
initilize_plt_conf(plt)

for c, folder in enumerate(result_folders):
    name_folder = folder.split("/")[-2]
    line_names.append(name_folder)
    min_y_axis=math.inf
    results = []
    files = glob(folder + "/*.pkl")
    x = get_x(files[0])
    for file_path in files:
        last_x,last_y = get_result_file(results, file_path)
        if (last_x < min_x_axis):
            min_x_axis = last_x
        if (last_y < min_y_axis):
            min_y_axis = last_y
    x=x[:min_y_axis]
    normalized_results=[]
    for data in results:
        normalized_results.append(data[:min_y_axis])
    mean = np.mean(np.array(normalized_results), axis=0)
    std = np.std(np.array(normalized_results), axis=0)
    plt.plot(x, mean, colors[c],label=name_folder)
    plt.fill_between(x, mean - std, mean + std, edgecolor=colors[c],
                     facecolor=colors[c], alpha=0.21,
                     linewidth=0)
    dict_results[name_folder] = {"mean": mean, "std": std}

plt.legend(loc='upper left')
axes = plt.gca()
plt.show()
