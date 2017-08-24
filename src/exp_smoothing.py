from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy import interpolate

import pandas as pd


def resolution_smoothing(strokes_list):
    distances = np.array([50 * x for x in range(1, 41)])
    # dist_strokes = np.vstack([distances, strokes_list]).T
    # smoothed_strokes_list = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dist_strokes)
    # score_samples = np.exp(kde.score_samples(dist_strokes)
    dist_smooth = np.linspace(distances.min(), distances.max(), 200)
    # smoothed_strokes_list = interpolate.spline(distances, strokes_list, dist_smooth)
    # smoothed_strokes_list = interpolate.UnivariateSpline(distances, strokes_list)(dist_smooth)
    smoothed_strokes_list = []
    for index, value in enumerate(strokes_list):
        if index > 0 and index < (len(strokes_list)-1):
            if strokes_list[index-1] == strokes_list[index+1]:
                # if abs(value-strokes_list[index+1]) == 1:
                value = np.mean([value,strokes_list[index+1]])
        smoothed_strokes_list.append(value)
    # smoothed_strokes_list = interpolate.spline(distances, smoothed_strokes_list, dist_smooth)
    # smoothed_strokes_list = interpolate.InterpolatedUnivariateSpline(distances, smoothed_strokes_list)(dist_smooth)
    return smoothed_strokes_list

def plot_strokes_list(strokes_list, smoothed_strokes, name):
    distances = [50 * x for x in range(1, 41)]
    colors = mpl.cm.rainbow(np.linspace(0, 1, 2))
    fig, ax = plt.subplots()
    ax.plot(distances, strokes_list, label='original', color=colors[0])
    ax.plot(distances, smoothed_strokes, label='smoothed', color=colors[1])
    plt.ylabel('gradient (strokes/minute)')
    plt.xlabel('distance (meters)')
    # Create a legend
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True)
    # legend = ax.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.savefig('../figures/smoothed_strokes/' + name + '.png')


def smooth_plot_strokes_list(strokes_list, name, plot_indicator):
    """ Read from the temperature sensor - and smooth the value out. The
    sensor is noisy, so we use exponential smoothing. """
    strokes_list = strokes_list.values
    # strokes_list = np.insert(strokes_list, 0, 0.0)
    smoothed_strokes = resolution_smoothing(strokes_list)
    if plot_indicator:
        plot_strokes_list(strokes_list, smoothed_strokes, name)
    return smoothed_strokes


