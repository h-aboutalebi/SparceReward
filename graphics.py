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
            self.min_x_number = 0
            self.max_x_number = len(example_dict)
        self.x_new_values, power_smooth = self.make_smooth_line([x['mod_reward'] for x in example_dict])
        return power_smooth

    # implementation from https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    def make_smooth_line(self, list):
        xnew = np.linspace(self.min_x_number, self.max_x_number,
                           len(list))  # 300 represents number of points to make between T.min and T.max
        # spl = make_interp_spline(self.x_values, list, k=55)  # BSpline object
        # power_smooth = spl(xnew)
        ysmoothed = gaussian_filter1d(list, sigma=1)
        return xnew, np.array(ysmoothed)

    def plot_figure(self, arr):
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        # plt.yscale('log')
        plt.xscale('log')
        fig = plt.figure()
        plt.plot(self.x_new_values, arr, "b", label=self.name)
        plt.legend(loc='upper left')
        fig.savefig(self.image_path)


# Create_Graph("/Users/hosseinaboutalebi/results_exploration_policy/2019-11-23_22:04:40.982046/results.pkl",
#              "/Users/hosseinaboutalebi/results_exploration_policy/2019-11-23_22:04:40.982046")
