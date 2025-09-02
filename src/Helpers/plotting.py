import os
from typing import List, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.size'] = 28


def plot_single_histogram(output_dir: str,
                          hist: np.ndarray,
                          legends: dict = None,
                          vline: bool = False,
                          pixels_before: np.array = None) -> None:
    """
    Plot histogram from data contained in hist

    :param output_dir: the directory where to save the files
    :type output_dir: str
    :param hist: the histogram data to plot
    :type hist: np.ndarray
    :param legends: the legend of the graph if any, defaults to None
    :type legends: dict, optional
    :param vline: a vertical line to plot, defaults to False
    :type vline: bool, optional
    :param pixels_before: the pixels before to plot also, defaults to None
    :type pixels_before: np.array, optional
    """

    fig = plt.figure()
    plt.plot(hist)

    plot = plot_legends(plt, legends, hist, vline, pixels_before)

    if "name" in legends.keys():
        plot.savefig(os.path.join(output_dir, legends["name"]+".png"))
    else:
        plot.savefig(os.path.join(output_dir, "no_name.png"))

    plot.close(fig)


def plot_multiple_histogram(output_dir: str,
                            hists: dict,
                            legends: dict = None,
                            vline: bool = False) -> None:
    """
    Plot multiple histograms from data contained in hists

    :param output_dir: the directory where to save the files
    :type output_dir: str
    :param hists: the histograms to plot the data
    :type hists: dict
    :param legends: the legend of the graph if any, defaults to None
    :type legends: dict, optional
    :param vline: the pixels before to plot also, defaults to False
    :type vline: bool, optional
    """

    fig = plt.figure()
    for key, val in hists.items():
        plt.plot(val, label=key)

    plot = plot_legends(plt, legends, val, vline)

    plot.xticks(fontsize=14)
    plot.yticks(fontsize=14)
    if "name" in legends.keys():
        plot.savefig(os.path.join(output_dir, legends["name"]+".png"))
    else:
        plot.savefig(os.path.join(output_dir, "no_name.png"))

    plot.close(fig)


def plot_bars(output_dir: str,
              separated_patches: dict = None,
              legends: dict = None) -> None:
    """
    Plot bars from good and bad patches detected

    :param output_dir: the directory where to save the files
    :type output_dir: str
    :param separated_patches: the separated patches, defaults to None
    :type separated_patches: dict, optional
    :param legends: the legend of the graph if any, defaults to None
    :type legends: dict, optional
    """
    height = [len(val) for val in list(separated_patches.values())]
    fig, ax = plt.subplots()
    bars = ax.bar(separated_patches.keys(), height)
    ax.bar_label(bars)
    plot = plot_legends(plt, legends)

    if "name" in legends.keys():
        plot.savefig(os.path.join(output_dir, legends["name"]+".png"))
    else:
        plot.savefig(os.path.join(output_dir, "no_name.png"))

    plot.close(fig)


def plot_legends(plot: plt,
                 legends: dict = None,
                 hist: np.array = None,
                 vline: bool = False,
                 pixels_before: np.array = None) -> plt:
    """
    Helper to plot legend to histogram

    :param plot: the plot to add leged to
    :type plot: plt
    :param legends: the legend to add, defaults to None
    :type legends: dict, optional
    :param hist: the histogram to add to the plot, defaults to None
    :type hist: np.array, optional
    :param vline: the vertical line to add, defaults to False
    :type vline: bool, optional
    :param pixels_before: the pixels before to add, defaults to None
    :type pixels_before: np.array, optional
    :return: the updated matplotlib.pyplot plot
    :rtype: plt
    """

    if vline:
        if pixels_before:
            hist = sum(np.asarray(
                list(pixels_before.values()), dtype=object)[:, 0])
        bins = np.linspace(0, 256, 257)
        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=hist)
        var = np.average((mids - mean)**2, weights=hist)
        max_hist = np.max(hist)
        plt.vlines(x=[mean-np.sqrt(var), mean+np.sqrt(var)],
                   ymin=0, ymax=max_hist, label="std", color="r")
        plt.vlines(x=[mean-var, mean+var], ymin=0,
                   ymax=max_hist, label="var", color="b")
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


def plot_otsu(cones_detected_hist: np.ndarray,
              threshold: float,
              analysis_dir: str) -> None:
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
    plt.savefig(os.path.join(analysis_dir, "detect_otsu_distribution.png"))
    plt.close("all")

# def display_and_save_image(
#         image: np.ndarray,
#         title: str,
#         save_fig: bool = False,
#         filename: str = None,
#         save_only: bool = False,
#         subject_num: int = None,
#         **kwargs
# ) -> None:
#     """
#     Plots the image with its title and saves it if necessary. Will pass additional keyword arguments to matplotlib.
#     If save_only is True, then the image will not be shown, but instantly saved
#     """

#     fig = plt.figure()
#     plt.imshow(image, **kwargs)
#     plt.title(title)

#     if save_fig:
#         filename = filename or title + '.png'
#         if subject_num is None:
#             plt.savefig(os.path.join(OUTPUT_DIR, filename))
#         else:
#             save_dir = OUTPUT_DIR
#             plt.savefig(os.path.join(save_dir, filename))

#         if save_only:
#             plt.close("all")
#             return
#     plt.tight_layout()
#     plt.show()

# def highlight_coordinates_on_image(
#         image: np.ndarray,
#         coordinates: np.ndarray,
#         marker_size: int = 3,
#         marker_color: np.ndarray = np.array([1., 0., 0.])
# ) -> np.ndarray:
#     """
#     Markers each given coordinate, superimposing squares of desired size and color on the image.
#     If the image is grayscale, its 3-channel copy will be used instead.

#     :param image: RGB or grayscale image
#     :param coordinates: (N, 2) array of pixel coordinates
#     :param marker_size: length of a marker edge
#     :param marker_color: (3, ) array of marker color;
#         ATTENTION: if its values are floats, then their maximum value is
#         considered 1.0, if they are integers, then the maximum value is 255
#     :return: an RGB image with given coordinates enveloped by the squares of given size and color
#     """

#     if np.issubdtype(coordinates.dtype, np.floating):
#         coordinates = np.asarray(np.floor(coordinates), dtype=PIXEL_COORDINATES_DATATYPE)

#     # create an rgb image looking the same as the given one
#     if len(image.shape) != 3:
#         resulting_image = np.empty((image.shape[0], image.shape[1], 3), dtype=image.dtype)
#         for i in range(3):
#             resulting_image[:, :, i] = image
#     else:
#         resulting_image = np.copy(image)

#     # prepare a mask with ones on the center coordinates only
#     mask = np.zeros(image.shape[:2])
#     mask[coordinates[:, 0], coordinates[:, 1]] = 1

#     # now mask will have square (marker_size x marker size) spots of value 1 with centers in given coordinates
#     if marker_size > 1:
#         marker = np.ones((marker_size, marker_size))
#         mask = signal.fftconvolve(mask, marker, 'same')

#     # convert the mask into boolean bitmap
#     mask = np.where(mask > 0.5, True, False)

#     # convert the marker color into the same (int in [0, 255] or float in [0, 1]) model as the original image
#     if np.issubdtype(marker_color.dtype, np.floating):
#         if np.issubdtype(image.dtype, np.integer):
#             marker_color = np.asarray(marker_color * 255, dtype=np.int32)
#     else:
#         if np.issubdtype(image.dtype, np.floating):
#             marker_color = np.asarray(marker_color / 255, dtype=np.float32)

#     # superimpose markers on the image
#     for i in range(3):
#         resulting_image[:, :, i] = np.where(mask, marker_color[i], resulting_image[:, :, i])

#     return resulting_image
