import os
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, output_path):
        self.output_path = output_path
        plt.rcParams['figure.figsize'] = (12, 12)
        plt.rcParams['font.size'] = 28

    def save_plot(self, plot, legends):
        if "name" in legends.keys():
            plot.savefig(os.path.join(self.output_path, legends["name"]+".png"))
        else:
            plot.savefig(os.path.join(self.output_path, "no_name.png"))
        plot.close("all")

    def plot_single_histogram(self, hist, legends=None, vline=False, pixels_before=None):
        fig = plt.figure()
        plt.plot(hist)
        plot = self.plot_legends(plt, legends, hist, vline, pixels_before)
        self.save_plot(plot, legends)

    def plot_multiple_histogram(self, hists, legends=None, vline=False):
        fig = plt.figure()
        for key, val in hists.items():
            plt.plot(val, label=key)
        plot = self.plot_legends(plt, legends, val, vline)
        self.save_plot(plot, legends)

    def plot_bars(self, separated_patches=None, legends=None):
        height = [len(val) for val in list(separated_patches.values())]
        fig, ax = plt.subplots()
        bars = ax.bar(separated_patches.keys(), height)
        ax.bar_label(bars)
        plot = self.plot_legends(plt, legends)
        self.save_plot(plot, legends)

    def plot_legends(self, plot, legends=None, hist=None, vline=False, pixels_before=None):
        if vline:
            if pixels_before:
                hist = sum(np.asarray(list(pixels_before.values()), dtype=object)[:, 0])
            bins = np.linspace(0, 256, 257)
            mids = 0.5*(bins[1:] + bins[:-1])
            mean = np.average(mids, weights=hist)
            var = np.average((mids - mean)**2, weights=hist)
            max_hist = np.max(hist)
            plt.vlines(x=[mean-np.sqrt(var), mean+np.sqrt(var)], ymin=0, ymax=max_hist, label="std", color="r")
            plt.vlines(x=[mean-var, mean+var], ymin=0, ymax=max_hist, label="var", color="b")
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

