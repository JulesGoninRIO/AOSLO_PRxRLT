from typing import Dict, List, Tuple
import os
import re
from pathlib import Path
import numpy as np
import cv2
import logging

from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
from skimage import morphology

from src.shared.datafile.helpers import ImageModalities

IMAGE_SIZE = 720


class DarkRegionFinder:
    pass

def find_dark_region_atms(output_path: str) -> Dict[str, np.array]:
    """
    Finds the spots of the dark region in mikhail's ATMS way of finding dark areas

    :param output_path: the path where the results from ATMS are
    :type output_path: str
    :return: boolean bitmap for each image where True means it is a dark pixel
    :rtype: Dict[str, np.array]
    """

    dark_regions = {}
    dark_regions_map = {}
    def condition_for_atms_dark_bitmap(name): return "bitmap" in name
    filenames = [name for name in os.listdir(
        output_path) if condition_for_atms_dark_bitmap(name)]

    for image_name in filenames:
        image_no_crop_name = re.sub(r"CROP_(\d+)_", "", image_name)
        image = cv2.imread(os.path.join(
            output_path, image_name), cv2.IMREAD_GRAYSCALE)
        image_number = re.search(r"_(\d+)_", image_no_crop_name).group(0)
        # get the position of the patch
        position = re.search(r"x(\d+)y(\d+)", image_name)
        x_position = eval(position.group(1))
        y_position = eval(position.group(2))
        # some images have a bitmap but too many dark regions, thus no cones detected
        # so we take the whole patch as being a dark area
        if image_name[:-11] + "_labeled.csv" in os.listdir(output_path):
            dark_region = np.argwhere(image == 0) + (y_position, x_position)
        else:
            dark_region = np.append(np.argwhere(image == 0) + (y_position, x_position),
                                    np.argwhere(image) + (y_position, x_position), axis=0)
        # add the dark region to the built dictionnary
        if image_number in dark_regions.keys():
            dark_regions[image_number] = np.append(
                dark_regions[image_number], dark_region, axis=0)
        else:
            dark_regions[image_number] = dark_region

    # creates a boolean bitmap from the dark regions per images
    for key, val in dark_regions.items():
        dark_map = np.full((IMAGE_SIZE, IMAGE_SIZE), False, dtype=bool)
        for point in val:
            dark_map[point[0], point[1]] = True
        dark_regions_map[key] = dark_map

    return dark_regions_map

def find_dark_region_mdrnn(output_path: str, image_numbers: List[str]) -> Dict[str, np.array]:
    """
    Finds the spots of the dark region in Davidson's MDRNN way of finding dark areas
    which is by adaptive thresholding (see separate_dark_regions)

    :param output_path: the path where the results from MDRNN are
    :type output_path: str
    :param image_numbers: the image numbers to detect dark regions on
    :type image_numbers: List[str]
    :return: boolean bitmap for each image where True means it is a dark pixel
    :rtype: Dict[str, np.array]
    """

    dark_regions_to_discard = {}
    for image_number in image_numbers:
        # let's find the Confocal images
        def condition_on_image(name): return image_number in name
        image_name = [name for name in os.listdir(Path(output_path).parent) if
                      condition_on_image(name)]
        try:
            image_name = [image for image in image_name if ImageModalities.CO.value
                          in image][0]
        except IndexError:
            logging.exception(f"The file {image_number} cannot be found in, \
                                {output_path}, make sure it exists.")
            continue
        image = cv2.imread(os.path.join(Path(output_path).parent, image_name),
                           cv2.IMREAD_GRAYSCALE)
        dark_regions = separate_dark_regions(image, image_name)
        # 1 = cells, 0 = dark areas -> need to invert that to have 1 = dark areas
        dark_regions_to_discard[image_number] = np.invert(dark_regions)

    return dark_regions_to_discard

def separate_dark_regions(image: np.ndarray, save_path: str = None) -> np.ndarray:
    """
    Adaptive dark region thresholding algorithm that works on Confocal images to
    detect dark areas that corresponds to vessels
    The steps are the following:
    1. Apply CLAHE with squared tile size equal to 4.
    2. Bilateral filtering of the image, with sigma = 100 and of diameter equal to 51.
    3. Gaussian filtering of the image, with sigma = 10 and of size 21.
    4. Normalization of the image.
    5. LOWESS the pixel values to have a smoothed histogram.
    6. Find all local maxima in the histogram values.
    7. Isolate the peaks in the left tail of the mean minus the standard
    deviation of the histogram bins, to get the central peak that has the
    biggest histogram value.
    8. Take the threshold to be where the biggest difference in first derivatives
    with a negative second derivative between points, ensuring the threshold is
    between the 25% quartiles of the center peak.
    9. Remove small unconnected components by applyingfive repeated dilations of
    the background followed by five erosions to remove small non-connected components
    (before) Marks all the regions where the pixel values are bigger than the average values. Before detection a high-sigma
    10. Gaussian blur is applied to detect not the individual cells, but the areas on the image where they are located.
    The detection assumes, that all the image values are distributed as a Gaussian or a mixture of two Gaussians,
    one of which corresponds to shadowed areas.

    :param image: grayscale image to threshold
    :type image: np.ndarray
    :param save_path: to save the binary map if needed, defaults to None
    :type save_path: str, optional
    :return: boolean mask indicating receptor-containing regions
             (and all the others are most likely to be blood vessels)
    :rtype: np.ndarray
    """

    # CLAHE equalization
    h, w = image.shape
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize=(4, 4))
    try:
        equalized = clahe.apply(image)
    except cv2.error:
        image = image.astype(np.uint8)
        equalized = clahe.apply(image)

    # Bilateral Filter and Gaussain Blur followed by normalization
    im_blur = cv2.bilateralFilter(equalized, 51, 100, 100)
    im_blur = cv2.GaussianBlur(im_blur, (21, 21), 10.0)
    im_blur = cv2.normalize(im_blur, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im_blur = im_blur*255
    im_blur = im_blur.astype(np.uint8)

    # Work on histogram of the pixel values then
    hist = np.histogram(im_blur, bins=len(np.unique(im_blur)), range=(0,255))
    values = hist[0]
    bins = 0.5*(hist[1][1:]+hist[1][:-1])
    try:
        smoothed = lowess(values, bins, is_sorted=False, frac=15/len(values), it=0)
        bins = smoothed[:,0]
        values = smoothed[:,1]
    except ValueError:
        # frac should be from 0 to 1 because the bins are too small
        if not len(bins)<2:
            smoothed = lowess(values, bins, is_sorted=False, frac=1, it=0)
            bins = smoothed[:,0]
            values = smoothed[:,1]
    indice = customized_peak_detection(bins, values)

    # threshold the image as grayscale
    try:
        threshold = (bins[indice]+bins[indice+1])/2
    except IndexError:
        # the bins array is too small
        threshold = bins[indice]
    binary_local = im_blur > threshold
    binary_local = remove_small_dark_dots(binary_local, size = 40)

    # # save the binary map if needed
    # if save_path:
    #     cv2.imwrite(os.path.join(r"C:\Users\BardetJ\Downloads", save_path), 255*binary_local.astype(np.uint8))
    return binary_local

def customized_peak_detection(bins: np.array, values: np.array) -> Tuple[int, int]:
    """
    Home-made Peak detection algorithm

    :param bins: the bins of the histogram
    :type bins: np.array
    :param values: the values of the histogram
    :type values: np.array
    :return: the indice of the peak
    :rtype: Tuple[int, int]
    """

    # Find the peaks in the histogram values
    peaks, _ = find_peaks(values)
    neg_peaks, neg_peaks_dic = find_peaks(-values, width=0)
    central_peaks = [indice for indice in peaks if bins[indice]>np.mean(bins)-np.std(bins)]
    try:
        central_peak = [indice for indice in central_peaks if values[indice] == \
                        np.max(values[central_peaks])][0]
    except IndexError:
        # the central peak has not been found, will use then the peak value found
        try:
            central_peak = peaks[0]
        except IndexError:
            # the peaks are empty, will only use the value
            central_peak = values[0]
    slopes = np.diff(values)/np.diff(bins)
    try:
        diff_slopes = np.array([j-i for i, j in zip(slopes[:central_peak-1], \
                                slopes[1:central_peak])])
        second_diff = [j-i for i, j in zip(diff_slopes[:-1], diff_slopes[1:])]
        second_diff = [(i+j)/2 for i, j in zip(second_diff[:-1], second_diff[1:])]
        min_found = False
        while not min_found:
            indice = np.argmax(diff_slopes)
            if indice+1 < len(second_diff) :
                if indice < 0.25*central_peak or indice > 0.75*central_peak:
                    diff_slopes[indice] = -np.inf
                else:
                    min_found = True
            else :
                diff_slopes[indice] = -np.inf
    except (TypeError, ValueError):
        # central peak array is too small, the indice found will be the mean of
        # the histogram bins
        indice = round(len(bins)/2)

    return indice

def remove_small_dark_dots(thresholded_image: np.ndarray, size: int = 10) -> np.ndarray:
    """
    Removes all the dark areas less then the given size from the given bitmap.

    :param thresholded_image: the image where dark dots will be removed
    :type thresholded_image: np.ndarray
    :param size: the size of the dots to suppress, defaults to 10
    :type size: int, optional
    :return: the image with no more dark dots
    :rtype: np.ndarray
    """

    for _ in range(size // 2):
        thresholded_image = morphology.binary_dilation(thresholded_image)
    for _ in range(size // 2):
        thresholded_image = morphology.binary_erosion(thresholded_image)

    return thresholded_image
