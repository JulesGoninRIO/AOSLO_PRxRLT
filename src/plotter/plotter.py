from typing import Dict
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import statsmodels.api as sm

class Plotter:
    def __init__(self, output_path: Path):
        """
        Initialize the Plotter.

        :param output_path: The path where the output will be saved.
        :type output_path: Path
        """
        self.output_path = output_path
        self.set_plot_styles()

    def set_plot_styles(self):
        """
        Set the plot styles using matplotlib parameters.
        """
        plt.style.use('ggplot')  # Use ggplot style for all plots
        plt.rcParams['figure.figsize'] = (10, 6)  # Default figure size
        plt.rcParams['figure.dpi'] = 300  # Default figure dpi
        plt.rcParams['font.size'] = 12  # Default font size
        plt.rcParams['lines.linewidth'] = 2  # Default line width
        plt.rcParams['axes.labelsize'] = 14  # Default label size
        plt.rcParams['axes.titlesize'] = 16  # Default title size
        plt.rcParams['xtick.labelsize'] = 12  # Default x-tick label size
        plt.rcParams['ytick.labelsize'] = 12  # Default y-tick label size
        plt.rcParams['legend.fontsize'] = 12  # Default legend font size
        plt.rcParams['figure.titlesize'] = 18  # Default figure title size

    def save_plot(self, plot: plt, legends: Dict[str, str]):
        """
        Save the plot to the specified output path.

        :param plot: The plot to save.
        :type plot: plt
        :param legends: The dictionary containing legend information.
        :type legends: Dict[str, str]
        """
        if "name" in legends.keys():
            plot.savefig(os.path.join(self.output_path, legends["name"] + ".png"))
        else:
            plot.savefig(os.path.join(self.output_path, "no_name.png"))
        plot.close("all")

    def plot_single_histogram(
        self,
        hist: np.ndarray,
        legends: Dict[str, str] = None,
        vline: bool = False,
        pixels_before: np.ndarray = None
        ):
        """
        Plot a single histogram.

        :param hist: The histogram data to plot.
        :type hist: np.ndarray
        :param legends: The dictionary containing legend information.
        :type legends: Dict[str, str], optional
        :param vline: Whether to draw vertical lines for mean and variance.
        :type vline: bool, optional
        :param pixels_before: The pixel data before the current state.
        :type pixels_before: np.ndarray, optional
        """
        fig = plt.figure()
        plt.plot(hist)
        plot = self.plot_legends(plt, legends, hist, vline, pixels_before)
        self.save_plot(plot, legends)

    def plot_multiple_histogram(
        self,
        hists: Dict[str, np.ndarray],
        legends: Dict[str, str] = None,
        vline: bool = False
        ):
        """
        Plot multiple histograms.

        :param hists: The dictionary containing multiple histograms.
        :type hists: Dict[str, np.ndarray]
        :param legends: The dictionary containing legend information.
        :type legends: Dict[str, str], optional
        :param vline: Whether to draw vertical lines for mean and variance.
        :type vline: bool, optional
        """
        fig = plt.figure()
        for key, val in hists.items():
            plt.plot(val, label=key)
        plot = self.plot_legends(plt, legends, val, vline)
        self.save_plot(plot, legends)

    def plot_bars(
        self,
        separated_patches: Dict[str, np.ndarray] = None,
        legends: Dict[str, str] = None
        ):
        """
        Plot bar charts for the given patches.

        :param separated_patches: The dictionary containing separated patches.
        :type separated_patches: Dict[str, np.ndarray], optional
        :param legends: The dictionary containing legend information.
        :type legends: Dict[str, str], optional
        """
        height = [len(val) for val in list(separated_patches.values())]
        fig, ax = plt.subplots()
        bars = ax.bar(separated_patches.keys(), height)
        ax.bar_label(bars)
        plot = self.plot_legends(plt, legends)
        self.save_plot(plot, legends)

    def plot_legends(
        self,
        plot: plt,
        legends: Dict[str, str] = None,
        hist: np.ndarray = None,
        vline: bool = False,
        pixels_before: np.ndarray = None
        ) -> plt:
        """
        Add legends and optional vertical lines to the plot.

        :param plot: The plot to add legends to.
        :type plot: plt
        :param legends: The dictionary containing legend information.
        :type legends: Dict[str, str], optional
        :param hist: The histogram data for calculating mean and variance.
        :type hist: np.ndarray, optional
        :param vline: Whether to draw vertical lines for mean and variance.
        :type vline: bool, optional
        :param pixels_before: The pixel data before the current state.
        :type pixels_before: np.ndarray, optional
        :return: The plot with legends added.
        :rtype: plt
        """
        if vline:
            if pixels_before:
                hist = sum(np.asarray(list(pixels_before.values()), dtype=object)[:, 0])
            bins = np.linspace(0, 256, 257)
            mids = 0.5 * (bins[1:] + bins[:-1])
            mean = np.average(mids, weights=hist)
            var = np.average((mids - mean) ** 2, weights=hist)
            max_hist = np.max(hist)
            plt.vlines(x=[mean - np.sqrt(var), mean + np.sqrt(var)], ymin=0, ymax=max_hist, label="std", color="r")
            plt.vlines(x=[mean - var, mean + var], ymin=0, ymax=max_hist, label="var", color="b")
            plt.vlines(x=mean, ymin=0, ymax=max_hist, label="mean", color="g")
            plt.legend()
        if "title" in legends.keys():
            plt.title(legends["title"], fontsize=14)
        if "xlabel" in legends.keys():
            plt.xlabel(legends["xlabel"], fontsize=14)
        if "ylabel" in legends.keys():
            plt.ylabel(legends["ylabel"], fontsize=14)
        if "write_legend" in legends.keys():
            plt.legend(prop={'size': 14})

        return plt
