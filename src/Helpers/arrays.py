import math
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_erosion, generate_binary_structure
from skimage.metrics import structural_similarity
from sklearn.metrics import normalized_mutual_info_score

from src.Helpers.Exceptions import EmptyVector, ZeroVector


def normalize(vector: np.array) -> np.array:
    """
    Normalize an array

    :param vector: 1d numpy array
    :type vector: np.array
    :return: vector divided by its L2 norm and zero if the values of the array
             are too small
    :rtype: np.array
    """

    norm = np.sqrt(np.sum(vector**2))
    if norm > 1e-3:
        return vector / norm
    else:
        return np.zeros_like(vector)


def zero_padding(src: np.ndarray, shape: Tuple[int, int], pos: np.array) -> np.ndarray:
    """
    Add zeros on the border of the images

    :param src: source image to pad
    :type src: np.ndarray
    :param shape: the shape to be padded to
    :type shape: Tuple[int, int]
    :param pos: the position y and x where the image will be
    :type pos: np.array
    :return: the image padded
    :rtype: np.ndarray
    """

    y, x = (int(pos[0]), int(pos[1]))
    padded_image = np.zeros(shape)
    padded_image[y:src.shape[0] + y, x:src.shape[1] + x] = src

    return padded_image


def to_shape(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Zero pad an image to match a certain shape
    https://stackoverflow.com/questions/56357039/numpy-zero-padding-to-match-a-certain-shape

    :param a: the image to pad
    :param shape: the shape to pad to

    :return: the image padded

    :param a: the image to pad
    :type a: np.ndarray
    :param shape: the shape to pad to
    :type shape: Tuple[int, int]
    :return: the image padded
    :rtype: np.ndarray
    """

    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)

    return np.pad(a, ((y_pad//2, y_pad//2 + y_pad % 2),
                      (x_pad//2, x_pad//2 + x_pad % 2)),
                  mode='constant')


def get_boundaries(dst: np.ndarray) -> List[int]:
    """
    Get the boundaries where the image pixels are

    :param dst: the image
    :return: list that has the form [top, bottom, left, right]

    :param dst: the image with the two modalities overlapping
    :type dst: np.ndarray
    :return: list that has the form [top, bottom, left, right]
    :rtype: List[int]
    """

    filled_pixels = np.argwhere(dst > 0)
    top = np.min(filled_pixels[:, 0])
    bottom = np.max(filled_pixels[:, 0])
    left = np.min(filled_pixels[:, 1])
    right = np.max(filled_pixels[:, 1])

    return [top, bottom, left, right]


def angle_between(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Get the angle between two points with the range ]-180:180]

    :param p1: the first point as (y,x)
    :type p1: Tuple[int, int]
    :param p2: the second point as (y,x)
    :type p2: Tuple[int, int]
    :return: a float corresponding to the angle between the points
    :rtype: float
    """

    return (math.atan2(p2[0]-p1[0], p2[1]-p1[1])*180)/math.pi



def detect_peaks(image: np.ndarray) -> np.array:
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    :param image: the image as an array
    :type image: np.ndarray
    :return: the peaks found in the image
    :rtype: np.array
    """

    # define an 8-connected neighborhood
    # neighborhood = generate_binary_structure(2,2)
    neighborhood = np.full((101, 101), True)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    try:
        local_max = maximum_filter(image, footprint=neighborhood) == image
    except ValueError:
        return 0
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def propagate(arr: np.array) -> np.array:
    """
    Propagate the array values by filling the None

    :param arr: the array to propagate
    :type arr: np.array
    :return: the array propagated
    :rtype: np.array
    """

    arg_values = np.argwhere(arr)
    for i in range(len(arg_values)):
        if i == len(arg_values)-1:
            for j in range(arg_values[i][0], len(arr)):
                arr[j] = arr[arg_values[i][0]]
        else:
            if i == 0:
                for j in range(arg_values[i][0]):
                    arr[j] = arr[arg_values[i][0]]
            difference_len = arg_values[i+1]-arg_values[i]
            difference_values = arr[arg_values[i+1][0]]-arr[arg_values[i][0]]
            increase_value = difference_values/difference_len
            for j in range(1, difference_len[0]):
                arr[arg_values[i][0] +
                    j] = round(arr[arg_values[i][0]]+j*increase_value[0])

    return arr
