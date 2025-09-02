
import csv
import glob
import os
import re
from pathlib import Path
from typing import List, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.PostProc_Pipe.Helpers.plotting import (plot_bars,
                                                plot_multiple_histogram,
                                                plot_single_histogram)

from .datafile_classes import CoordinatesFile
from .global_helpers import pixels_per_patch

# SHOULD DO A CLASS WITH OUTPUT DIR THE SAME EVERYWHERE


def preprocessing_plots(output_dir: str,
                        writing_dir: str,
                        pixels: Dict[str, np.array],
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
    plot_single_histogram(writing_dir, overall_hist,
                          legends, True, pixels_before)

    # Plot those values as an histogram
    legends = {'title': "Number of cones detected per Patch",
               'xlabel': "Number of cones",
               'ylabel': "Occurence",
               'name': "cones_detected_per_patch"}
    plot_single_histogram(writing_dir, np.asarray(
        cones_detected_hist[0]), legends, False)
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
    plot_multiple_histogram(writing_dir, hists, legends, vline=False)

    legends = {
        'title': "Number of working / non-working patches",
        'ylabel': "Occurence",
        'name': "num_good_bad_patches"
    }
    plot_bars(writing_dir, separated_patches, legends)

    # Let's investigate now good patches
    # Insert NST analysis on good patches here to look at difference in cone centers


def hist_matching(bad_image: np.array,
                  good_hist: np.array,
                  output_dir: str = None,
                  image_name: str = None) -> np.array:
    """
    Method to match histograms from bad patches to good one. Bad means few number
    of cones detected by the MDRNN algorithm

    :param bad_image: the bad images that have to be matches
    :type bad_image: np.array
    :param good_hist: the good patch to match the bad one to
    :type good_hist: np.array
    :param output_dir: the output directory to save matches, defaults to None
    :type output_dir: str, optional
    :param image_name: the name of the image, defaults to None
    :type image_name: str, optional
    :return: the updated bad patch that has been matches
    :rtype: np.array
    """

    # Look for output_dir and choose right one
    # fig_dir = Path(output_dir).parent.absolute()
    fig_dir = os.path.join(output_dir, "hist_matching")
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    bad_hist = np.histogram(bad_image.flatten(), 256, [0, 256])

    lookup_table = match_histograms(bad_hist, good_hist)
    # Use the lookup function to transform the colors of the original
    # source image
    after_transform = cv2.LUT(bad_image, lookup_table)

    # Put the image back together
    # image_after_matching = cv2.merge([
    #    blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(after_transform)

    match_hist = np.histogram(image_after_matching.flatten(), 256, [0, 256])
    hists = {'Not Working histogram': bad_hist[0],
             'Working histogram': good_hist, 'Matched histogram': match_hist}
    legends = {'xlabel': "Image pixels [0:255]",
               'ylabel': "Number of pixels",
               'name': f"histogram_matching_{image_name}",
               "write_legend": True}
    plot_multiple_histogram(fig_dir, hists, legends)

    return image_after_matching


def match_histograms(src_hist: np.array, ref_hist: np.array) -> np.array:
    """
    This method matches the source image histogram to the reference signal

    :param src_hist: The original source image
    :type src_hist: np.array
    :param ref_hist: The reference image
    :type ref_hist: np.array
    :return: image_after_matching
    :rtype: np.array
    """

    # Compute the normalized cdf for the source and reference image
    src_cdf = calculate_cdf(src_hist)
    ref_cdf = calculate_cdf(ref_hist)

    # Make a separate lookup table for each color
    lookup_table = calculate_lookup(src_cdf, ref_cdf)

    return lookup_table


def calculate_cdf(histogram: np.array) -> np.array:
    """
    This method calculates the cumulative distribution function

    :param histogram: the values of the histogram
    :type histogram: np.array
    :return: the normalized cumulative distribution function
    :rtype: np.array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf: np.array, ref_cdf: np.array) -> np.array:
    """
    This method creates the lookup table

    :param src_cdf: The cdf for the source image
    :type src_cdf: np.array
    :param ref_cdf: The cdf for the reference image
    :type ref_cdf: np.array
    :return: The lookup table
    :rtype: np.array
    """

    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val

    return lookup_table


def cones_histogram(cones_detected: Dict[str, float]) -> np.array:
    """
    Do the histogram of an the numebr of cones detected per patch

    :param cones_detected: the number of of cones detected per patch
    :type cones_detected: ict[str, float]
    :return: the histogram of the number of cones detected per patch
    :rtype: np.array
    """

    number_of_cones_per_patch = list(cones_detected.values())
    max_of_cones_per_patch = np.max(number_of_cones_per_patch)
    # The histogram of cones detected per patch
    cones_detected_hist = np.histogram(number_of_cones_per_patch,
                                       max_of_cones_per_patch,
                                       [0, max_of_cones_per_patch])

    return cones_detected_hist
