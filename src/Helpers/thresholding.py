import math

# newest versions of OpenCV changed the way of importing
try:
    from cv2 import cv2
except ImportError:
    import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple

from numba import njit, prange
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess

from skimage import morphology

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

@njit(cache=True)
def _threshold_grayscale_image(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies a given fixed threshold on the image.

    :param image: the image as a numpy ndarray
    :param threshold: the threshold to cut the pixel values
    :return: boolean mask containing False where the original pixel values are
        less than the threshold and True otherwise

    :param image: the image as a numpy ndarray
    :type image: np.ndarray
    :param threshold: the threshold to cut the pixel values
    :type threshold: float
    :return: boolean mask containing False where the original pixel values are
             less than the threshold and True otherwise
    :rtype: np.ndarray
    """

    return image > threshold

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

# def adaptive_thresh(input_img: np.ndarray):
#     """
#     Adaptive thresholding algorithm based on Bradley's Adaptive Thresholding
#     using the integral image
#     From https://stackoverflow.com/questions/29502241/bradley-adaptive-thresholding-algorithm

#     :return input_img: the image as a numpy ndarray
#     """

#     h, w = input_img.shape
#     S = w/4
#     s2 = S/2
#     T = 15.0

#     #integral img
#     int_img = np.zeros_like(input_img, dtype=np.uint64)
#     for col in range(w):
#         for row in range(h):
#             int_img[row,col] = input_img[0:row,0:col].sum()

#     #output img
#     out_img = np.zeros_like(input_img)

#     for col in range(w):
#         for row in range(h):
#             #SxS region
#             y0 = max(round(row-s2), 0)
#             y1 = min(round(row+s2), h-1)
#             x0 = max(round(col-s2), 0)
#             x1 = min(round(col+s2), w-1)

#             count = (y1-y0)*(x1-x0)

#             sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]

#             if input_img[row, col]*count < sum_*(100.-T)/100.:
#                 out_img[row,col] = 0
#             else:
#                 out_img[row,col] = 255

#     return out_img

# def separate_dark_regions_test(image: np.ndarray, name: str= None) -> np.ndarray:
#     """
#     Adaptive dark region thresholding algorithm

#     :param image: the image as a numpy ndarray
#     :param name: the name of the image to save the histograms of the pixel
#                  intensities.


#     """
#     # image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     # image = image*255
#     # image = image.astype(np.uint8)

#     # Normalize is useless before CLAHE or Hist eq
#     # CLAHE is better at rendering the image bright everywhere the same

#     # CLAHE
#     h, w = image.shape
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
#     equalized = clahe.apply(image)
#     # equ = cv2.equalizeHist(image)

#     im_blur = cv2.bilateralFilter(equalized, 51, 100, 100)
#     im_blur = cv2.GaussianBlur(im_blur, (21, 21), 10.0)

#     im_blur = cv2.normalize(im_blur, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     im_blur = im_blur*255
#     im_blur = im_blur.astype(np.uint8)

#     hist = np.histogram(im_blur, bins=len(np.unique(im_blur)), range=(0,255))

#     # Test sato
#     from skimage.filters import sato
#     im_out = sato(image, sigmas = range(40,80,20))
#     cv2.imshow("test", im_blur)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imshow("test", im_out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # from scipy.signal import find_peaks
#     # values = hist[0]
#     # # bins = hist[1]
#     # bins = 0.5*(hist[1][1:]+hist[1][:-1])

#     # from statsmodels.nonparametric.smoothers_lowess import lowess
#     # smoothed = lowess(values, bins, is_sorted=False, frac=15/len(values), it=0)
#     # bins = smoothed[:,0]
#     # values = smoothed[:,1]

#     # not_found = True

#     # peaks, _ = find_peaks(values)
#     # neg_peaks, neg_peaks_dic = find_peaks(-values, width=0)

#     # central_peaks = [indice for indice in peaks if bins[indice]>np.mean(bins)-np.std(bins)]
#     # central_peak = [indice for indice in central_peaks if values[indice] == np.max(values[central_peaks])][0]
#     # # good_peak = [indice for indice in neg_peaks if indice < central_peak and values[indice] > np.median(values)]
#     # # if good_peak :
#     # #     indices = [np.where(neg_peaks == peak)[0] for peak in good_peak]
#     # #     widths = [neg_peaks_dic["widths"][indice[0]] for indice in indices]
#     # #     indice = neg_peaks[indices[np.argmax(widths)]][0]
#     # # else :
#     # slopes = np.diff(values)/np.diff(bins)
#     # diff_slopes = np.array([j-i for i, j in zip(slopes[:central_peak-1], slopes[1:central_peak])])
#     # second_diff = [j-i for i, j in zip(diff_slopes[:-1], diff_slopes[1:])]
#     # second_diff = [(i+j)/2 for i, j in zip(second_diff[:-1], second_diff[1:])]
#     # min_found = False
#     # while not min_found:
#     #     indice = np.argmax(diff_slopes)
#     #     if indice+1 < len(second_diff) :
#     #         if indice < 0.25*central_peak or indice > 0.75*central_peak:
#     #             diff_slopes[indice] = 0
#     #         else:
#     #             min_found = True
#     #     else :
#     #         diff_slopes[indice] = 0

#     # threshold = (bins[indice]+bins[indice+1])/2
#     # binary_local = im_blur > threshold

#     # binary_local = remove_small_dark_dots(binary_local, size = 40)
#     # binary_local = binary_local.astype(np.uint8)*255

#     # plt.plot(bins, values)
#     # plt.axvline(x=threshold, color = "r", linewidth = 2)
#     # plt.xlabel("Pixel values [0,255]")
#     # plt.ylabel("Occurence")
#     # plt.savefig(os.path.join(r"C:\Users\BardetJ\Documents\tests\algo_steps\clahe_bilateral_gauss_normalize_algo_small_40", f"{name}_lowess.png"))
#     # plt.close("all")
#     # cv2.imwrite(os.path.join(r"C:\Users\BardetJ\Documents\tests\algo_steps\clahe_bilateral_gauss_normalize_algo_small_40", f"{name}_.png"), im_blur.astype(np.uint8))
#     # # plt.imshow(binary_local.astype(np.uint8))
#     # # plt.savefig(os.path.join(r"C:\Users\BardetJ\Documents\tests\algo_steps\clahe_bilateral_gauss_normalize_algo_small_40", f"{name}.png"))
#     # cv2.imwrite(os.path.join(r"C:\Users\BardetJ\Documents\tests\algo_steps\clahe_bilateral_gauss_normalize_algo_small_40", f"{name}.png"), binary_local.astype(np.uint8))
#     # plt.close("all")
#     # # cv2.imwrite(os.path.join(r"C:\Users\BardetJ\Documents\tests\algo_steps\clahe_bilateral_gauss_normalize_algo", f"{name}.png"), binary_local.astype(np.uint8))
