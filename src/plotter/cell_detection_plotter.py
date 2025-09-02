from typing import Dict, List
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.plotter.plotter import Plotter

class CellDetectionPlotter(Plotter):
    def __init__(self, output_path: Path):
        super().__init__(output_path)

    def plot_otsu_threshold(self, cones_detected_hist: np.ndarray, threshold: float) -> None:
        """
        Plot the OTSU thresholding on an histogram

        :param cones_detected_hist: the histogram with the number of cones detected
                                    per patch
        :type cones_detected_hist: np.ndarray
        :param threshold: the OTSU threshold
        :type threshold: float
        :param analysis_dir: the directory where to save the plot
        :type analysis_dir: str
        """

        x = np.linspace(0, 255, len(cones_detected_hist[0]))
        plt.plot(x, cones_detected_hist[0])
        plt.axvline(x=threshold, color="r")
        plt.xlabel("Number of cones detected per patch", fontsize=15)
        plt.ylabel("Occurence", fontsize=15)
        plt.title("OTSU threshold found", fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(str(self.output_path / "detect_otsu_distribution.png"))
        plt.close("all")

    def preprocessing_plots(self, pixels: Dict[str, np.array],
                                cones_detected_hist: np.array,
                                separated_patches: Dict[str, str],
                                pixels_before: np.array = None) -> None:
        """
        Function to save all the preprocessing plots on data

        :param output_dir: the output directory where the patches are
        :type output_dir: str
        :param writing_dir: the output directory where to save resulting graphs
        :type writing_dir: str
        :param pixels: the pixel distributions of the patches
        :type pixels: Dict[str, np.array]
        :param cones_detected_hist: the histogram of the number of cones detected
                                    per patches
        :type cones_detected_hist: np.array
        :param separated_patches: the patches separated as "good" patch (lot of cones
                                detected) or "bad" patches (view)
        :type separated_patches: Dict[str, str]
        :param pixels_before: the pixels before we have applied thresholding to the
                            histogram of the pixels, defaults to None
        :type pixels_before: np.array, optional
        """

        # First we plot the histogram of the sum of pixels values
        # For that let's look whether the histogram comes from np.histogram (tuple)
        # or cv2.calcHist (nd.array)
        if type(list(pixels.values())[0]) is tuple:
            val = np.asarray(list(pixels.values()), dtype=object)[:, 0]
            bins = np.asarray(list(pixels.values()), dtype=object)[:, 1]
        else:
            val = pixels.values()
        overall_hist = sum(val)
        legends = {'title': "Occurence of pixel values",
                'xlabel': "Image pixels [0:255]",
                'ylabel': "Number of pixels",
                'name': "cones_histograms_with_locations_and_std_var"}
        super().plot_single_histogram(overall_hist, legends, True, pixels_before)

        # Plot those values as an histogram
        legends = {'title': "Number of cones detected per Patch",
                'xlabel': "Number of cones",
                'ylabel': "Occurence",
                'name': "cones_detected_per_patch"}
        super().plot_single_histogram(np.asarray(cones_detected_hist[0]), legends, False)
        # plot_single_histogram(writing_dir, np.asarray(
        #     cones_detected_hist)[0], legends, False)

        # Let's now output plots from the separated patches
        good_hist = np.zeros([256,])
        for key in separated_patches["good"]:
            good_hist += pixels[key][0]
        good_hist = good_hist/len(separated_patches["good"])

        bad_hist = np.zeros([256,])
        for key in separated_patches["bad"]:
            bad_hist += pixels[key][0]
        bad_hist = bad_hist/len(separated_patches["bad"])
        hists = {"Working histogram": good_hist, "Not Working histogram": bad_hist}
        legends = {'title': "Histogram of pixel values",
                'xlabel': "Pixel values",
                'ylabel': "Occurence",
                'write_legend': "",
                'name': "cones_detected_per_separated_patch"}
        super().plot_multiple_histogram(hists, legends, vline=False)

        legends = {
            'title': "Number of working / non-working patches",
            'ylabel': "Occurence",
            'name': "num_good_bad_patches"
        }
        super().plot_bars(separated_patches, legends)

