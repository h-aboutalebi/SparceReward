import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import sem
from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline


class Create_Graph:
    def __init__(self, path_pkl, path_image,name):
        self.image_path = path_image + "/image.png"
        self.name=name
        results = self.get_results_from_file(path=path_pkl)
        self.plot_figure(results)

    def get_results_from_file(self, path):
        with open(path, 'rb') as f:
            example_dict = pickle.load(f)
        self.x_new_values=[x['step_nb'] for x in example_dict]
        power_smooth = self.make_smooth_line([x['mod_reward'] for x in example_dict])
        return power_smooth

    # implementation from https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    def make_smooth_line(self, list):
        ysmoothed = gaussian_filter1d(list, sigma=1)
        return np.array(ysmoothed)

    def plot_figure(self, arr):
        # plt.yscale('log')
        plt.tight_layout()
        fig = plt.figure()
        plt.xlabel('Number of steps')
        plt.ylabel('Reward')
        # plt.xscale('log')
        plt.plot(self.x_new_values, arr, "b", label=self.name)
        plt.legend(loc='upper left')
        fig.savefig(self.image_path, bbox_inches="tight")


# Create_Graph("/Users/hosseinaboutalebi/results_exploration_policy/2019-11-23_22:04:40.982046/results.pkl",
#              "/Users/hosseinaboutalebi/results_exploration_policy/2019-11-23_22:04:40.982046")
