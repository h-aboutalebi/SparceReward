import pickle
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from glob import glob
import math

def create_graph(plt, target, plt_figure, y_label, x_label, result_folders, colors,
                 folder_name_cons="",smoothness=5):
    dict_results = {}
    line_names = []
    initilize_plt_conf(plt,y_label,x_label)
    min_x_axis = math.inf
    for c, folder in enumerate(result_folders):
        name_folder = folder.split("/")[-2]
        if(folder_name_cons not in name_folder.lower()):
            continue
        line_names.append(name_folder)
        min_y_axis = math.inf
        results = []
        files = glob(folder + "/*.pkl")
        x = get_x(files[0])
        for file_path in files:
            last_x, last_y = get_result_file(results, file_path,target,smoothness=smoothness)
            if (last_x < min_x_axis):
                min_x_axis = last_x
            if (last_y < min_y_axis):
                min_y_axis = last_y
        x = x[:min_y_axis]
        normalized_results = []
        for data in results:
            normalized_results.append(data[:min_y_axis])
        mean = np.mean(np.array(normalized_results), axis=0)
        std = np.std(np.array(normalized_results), axis=0)
        plt.plot(x, mean, colors[c], label=name_folder)
        plt.fill_between(x, mean - std, mean + std, edgecolor=colors[c],
                         facecolor=colors[c], alpha=0.21,
                         linewidth=0)
        dict_results[name_folder] = {"mean": mean, "std": std}
    plt.legend(loc='upper left')
    axes = plt.gca()
    plt_figure.show()

def get_x(path_file):
    with open(path_file, 'rb') as f:
        example_dict = pickle.load(f)
    x = [x["step_nb"] for x in example_dict]
    return x

def get_result_file(results,path_file,target,smoothness=5):
    with open(path_file, 'rb') as f:
        example_dict = pickle.load(f)
    y_values=[x[target] for x in example_dict]
    if(smoothness!=0):
        y=make_smooth_line(y_values,smoothness)
    else:
        y=y_values
    results.append(y)
    return max_step_nb(example_dict),len(y_values)

def make_smooth_line(list,smoothness):
    ysmoothed = gaussian_filter1d(list, sigma=smoothness)
    return ysmoothed

def max_step_nb(example_dict):
    return (example_dict[-1]['step_nb']-example_dict[-2]['step_nb']-1)*(len(example_dict)+1)

def initilize_plt_conf(plt,y_label,x_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


