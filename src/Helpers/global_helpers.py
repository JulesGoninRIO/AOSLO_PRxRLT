from src.PostProc_Pipe.Helpers.global_constants import (
    ATMS_OVERLAP_DISTANCE, IMAGE_SIZE, MDRNN_OVERLAP_DISTANCE,
    MDRNN_OVERLAP_DISTANCE_SAME_IMAGE, NIEGHBOR_CIRCULAR_SHAPE)
from src.PostProc_Pipe.CellDetection.MDRNN.constants import RAW_DATA_DIR_NAME
from src.PostProc_Pipe.Helpers.datafile_classes import (
    CoordinatesFile, ImageFile, ImageModalities,
    select_files_with_given_parameters)
from src.PostProc_Pipe.Helpers.arrays import (angle_between, get_boundaries)
from src.PostProc_Pipe.Configs.Parser import Parser
from src.PostProc_Pipe.CellDetection.MDRNN.constants import \
    MDRNN_CONE_DETECTOR_OUT_DIR
from src.PostProc_Pipe.CellDetection.ComputerVisionEngine.global_names import \
    FOVEA_BORDER
import sys
import collections
import csv
import glob
import logging
import math
import os
import pickle
import re
import shutil
import timeit
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from numba import njit
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from .thresholding import separate_dark_regions

matplotlib.use('Agg')

# def find_points(image: np.ndarray, angle_c: int = 0) -> List[List[int]]:
#     """
#     Find the corners of an image and returns them as [top left, top right,
#     bottom left, bottom right]. We have to make sure the image does contain only
#     ones or has an alpha chanel, otherwise we can have zeros pixels that belong
#     to the image but is not recongized as such.

#     :param image: the image to find corners
#     :type image: np.ndarray
#     :raises AssertionError: the image has no alpha channel or is not composed of
#                             only zeros and ones
#     :return: the corners of the image
#     :rtype: List[List[int]]
#     """

#     base_top = 0
#     base_bottom = IMAGE_SIZE-1
#     base_left = 0
#     base_right = IMAGE_SIZE-1

#     # sometimes there are dark pixels in the contour of the image and it messes
#     # up the shift recognition so we add a white contour while keeping the image
#     # size constant so that the shifts find the right contours

#     if len(image.shape) > 2:
#         # we have an alpha channel so we will find corners with it
#         image = image[:,:,-1]
#     assert len(np.unique(image) == 2), "the image \
#         has no alpha channel or is not composed of only 2 values"

#     filled_pixels = np.argwhere(image > 0)
#     top, bottom, left, right = get_boundaries(image)

#     base_first_point = [base_top, base_left]
#     base_second_point = [base_top, base_right]
#     base_third_point = [base_bottom, base_left]
#     base_fourth_point = [base_bottom, base_right]

#     # Is it a top left or a top right corner ?
#     dist_top_left = abs(
#         np.min(filled_pixels[filled_pixels[:, 0] == top][:, 1]) - left)
#     dist_top_right = abs(
#         np.max(filled_pixels[filled_pixels[:, 0] == top][:, 1]) - right)
#     if dist_top_left < dist_top_right:
#         # top_loc is on the top left corner -> bottom has to be on the right
#         # rotation with PIL Image is negative
#         top_loc = np.min(filled_pixels[filled_pixels[:, 0] == top][:, 1])
#         first_point = [top, top_loc]
#         bottom_loc = np.max(filled_pixels[filled_pixels[:, 0] == bottom][:, 1])
#         fourth_point = [bottom, bottom_loc]
#         # and top right is the rightmost point
#         top_right_x = np.min(filled_pixels[filled_pixels[:, 1] == right][:, 0])
#         second_point = [top_right_x, right]
#         bottom_left_x = np.max(
#             filled_pixels[filled_pixels[:, 1] == left][:, 0])
#         third_point = [bottom_left_x, left]
#         # TODO: should be tested before modified
#         # if 45 <= angle_c <= 90:
#         #     first_point, second_point, third_point, fourth_point = third_point, fourth_point, second_point, first_point
#         # elif 90 < angle_c <= 180:
#         #     first_point, second_point, third_point, fourth_point = fourth_point, first_point, second_point, third_point
#         # elif 180 < angle_c <= 270:
#         #     first_point, second_point, third_point, fourth_point = second_point, third_point, fourth_point, first_point
#         # elif 270 < angle_c <= 360:
#         #     first_point, second_point, third_point, fourth_point = first_point, second_point, third_point, fourth_point
#         # elif -90 <= angle_c <= -45:
#         #     first_point, second_point, third_point, fourth_point = second_point, third_point, fourth_point, first_point
#         # elif -180 <= angle_c <= -90:
#         #     first_point, second_point, third_point, fourth_point = third_point, fourth_point, first_point, second_point
#         # elif -270 <= angle_c <= -180:
#         #     first_point, second_point, third_point, fourth_point = fourth_point, first_point, second_point, third_point
#         # elif -360 <= angle_c <= -270:
#         #     first_point, second_point, third_point, fourth_point = first_point, second_point, third_point, fourth_point
#         if 45 <= angle_c <= 90:
#             anciant_first = first_point
#             first_point = third_point
#             third_point = fourth_point
#             fourth_point = second_point
#             second_point = anciant_first
#         elif 90 <= angle_c <= 180:
#             first_point, second_point, third_point, fourth_point = third_point, fourth_point, first_point, second_point
#         elif 180 < angle_c <= 360:
#             import pdb
#             pdb.set_trace()
#         elif -90 <= angle_c <= -45:
#             import pdb
#             pdb.set_trace()
#         elif -180 <= angle_c <= -90:
#             anciant_first = first_point
#             first_point = second_point
#             second_point = fourth_point
#             fourth_point = third_point
#             third_point = anciant_first
#         elif -360 <= angle_c <= -180:
#             import pdb
#             pdb.set_trace()
#     else:
#         # top_loc is on the top right corner -> bottom has to be on the left
#         # rotation with PIL Image is positive
#         base_second_point = [base_top, base_right]
#         top_loc = np.max(filled_pixels[filled_pixels[:, 0] == top][:, 1])
#         second_point = [top, top_loc]
#         bottom_loc = np.min(filled_pixels[filled_pixels[:, 0] == bottom][:, 1])
#         third_point = [bottom, bottom_loc]
#         base_third_point = [base_bottom, base_left]
#         top_left_x = np.min(filled_pixels[filled_pixels[:, 1] == left][:, 0])
#         base_first_point = [base_top, base_left]
#         first_point = [top_left_x, left]
#         base_fourth_point = [base_bottom, base_right]
#         bottom_right_x = np.max(
#             filled_pixels[filled_pixels[:, 1] == right][:, 0])
#         fourth_point = [bottom_right_x, right]
#         # TODO: should be tested before modified
#         # if 45 <= angle_c <= 90:
#         #     first_point, second_point, third_point, fourth_point = fourth_point, first_point, second_point, third_point
#         # elif 90 < angle_c <= 180:
#         #     first_point, second_point, third_point, fourth_point = third_point, fourth_point, first_point, second_point
#         # elif 180 < angle_c <= 270:
#         #     first_point, second_point, third_point, fourth_point = second_point, third_point, fourth_point, first_point
#         # elif 270 < angle_c <= 360:
#         #     first_point, second_point, third_point, fourth_point = first_point, second_point, third_point, fourth_point
#         # elif -90 <= angle_c <= -45:
#         #     first_point, second_point, third_point, fourth_point = third_point, fourth_point, first_point, second_point
#         # elif -180 <= angle_c <= -90:
#         #     first_point, second_point, third_point, fourth_point = fourth_point, first_point, second_point, third_point
#         # elif -270 <= angle_c <= -180:
#         #     first_point, second_point, third_point, fourth_point = first_point, second_point, third_point, fourth_point
#         # elif -360 <= angle_c <= -270:
#         #     first_point, second_point, third_point, fourth_point = second_point, third_point, fourth_point, first_point
#         if 45 <= angle_c <= 90:
#             import pdb
#             pdb.set_trace()
#         elif 90 <= angle_c <= 180:
#             anciant_first = first_point
#             first_point = third_point
#             third_point = fourth_point
#             fourth_point = second_point
#             second_point = anciant_first
#         elif -90 <= angle_c <= -45:
#             anciant_first = first_point
#             first_point = second_point
#             second_point = fourth_point
#             fourth_point = third_point
#             third_point = anciant_first
#         elif -180 <= angle_c <= -90:
#             import pdb
#             pdb.set_trace()
#         elif -360 <= angle_c <= -180:
#             import pdb
#             pdb.set_trace()
#         elif 180 < angle_c <= 360:
#             import pdb
#             pdb.set_trace()

#     # points as top left, top right, bottom left, bottom right
#     base_points = [base_first_point, base_second_point,
#                    base_third_point, base_fourth_point]
#     points = [first_point, second_point, third_point, fourth_point]


#     return points


# def get_shift_from_points(base_points: List[int], points: List[int]) -> np.ndarray:
#     """
#     Get the affine transform from corners points of the image

#     :param base_points: the base corners of the image
#     :type base_points: List[int]
#     :param points: the transformed corners of the same image
#     :type points: List[int]
#     :return: the shift from the transformed image to the base one
#     :rtype: np.ndarray
#     """

#     # returns dy, dx
#     assert len(base_points) == len(points) == 3, "the points to find the shift "\
#         "transform don't have the correct dimensions, they should be 3 datapoints "\
#         "in each."
#     shift = cv2.getAffineTransform(np.float32(base_points), np.float32(points))
#     try:
#         assert np.isclose(abs(np.arccos(shift[0, 0])), abs(np.arccos(shift[1, 1])),
#                           atol=0.001)
#         assert np.isclose(abs(np.arcsin(shift[0, 1])), abs(np.arcsin(shift[1, 0])),
#                           atol=0.001)
#     except AssertionError:
#         logging.error("The shift of the image does not only contain translation "
#                       "and rotation but also scale shear or reflection.")
#         raise

#     return shift


# def create_own_shifts(input_dir: str, modality: ImageModalities = ImageModalities.CO) \
#         -> Dict[str, np.array]:
#     """
#     Returns the shifts of each image from a montage with the form
#     [[1,0,dy][0,1,dx]] because MATLAB montage's shifts are wrong and points as (y,x)

#     :param input_dir: directory where the montage images are
#     :type input_dir: str
#     :param modality: the modality to find the shift from, defaults to ImageModalities.CO
#     :type modality: ImageModalities, optional
#     :return: a dictionnary with the image names as key and the 2x3 matrix as
#              the shift of the image
#     :rtype: Dict[str, np.array]
#     """

#     # Let's gather the single images from a specific modality
#     filenames = {entry.name for entry in os.scandir(input_dir) if modality.value in entry.name}
#     # Avoid repetitive elements
#     filenames = set(filenames)

#     # Compile regular expressions for speed
#     regex1 = re.compile(r"_aligned_to_ref(\d+)_m(\d+)")
#     regex2 = re.compile("CalculatedSplit")

#     # We will need to find each shifts from the single images that have all the
#     # same size
#     dic_list = []
#     for image_name in filenames:
#         image = tifffile.imread(os.path.join(input_dir, image_name))
#         try:
#             points = find_points(image)
#         except AssertionError:
#             continue
#         out_name = regex2.sub("Confocal", regex1.sub("", image_name))
#         dic_data = {"name": out_name,
#                     "top_left_y": points[0][0],
#                     "top_left_x": points[0][1],
#                     "top_right_y": points[1][0],
#                     "top_right_x": points[1][1],
#                     "bottom_left_y": points[2][0],
#                     "bottom_left_x": points[2][1],
#                     "bottom_right_y": points[3][0],
#                     "bottom_right_x": points[3][1]}
#         dic_list.append(dic_data)

#     # If you want ot get the shifts from the entire connected compoenent as well
#     # we are gathering their shifts only from translation because the whole
#     # component has no rotation
#     shifts = pd.DataFrame.from_dict(dic_list)

#     return shifts


def get_points(shift: List[np.ndarray]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Get the points of the top left and top right corner from the shifts

    :param shift: the shift of an image
    :type shift: List[np.ndarray]
    :return: the top left and top right corners of the image
    :rtype: Tuple[Tuple[int, int], Tuple[int, int]]
    """

    logging.info("Get the points")

    return shift[0][0], shift[0][1]


def shift_image_from_points(shift: List[np.ndarray], image: np.ndarray = None) -> np.ndarray:
    """
    Shift an image based on the the points found on an array. It is use as
    cv2.warpAffine but without the bugs found there.

    :param shift: the shifts of the points
    :type shift: List[np.ndarray]
    :param image: the image to shift, defaults to None
    :type image: np.ndarray, optional
    :return: the image shifted
    :rtype: np.ndarray
    """

    logging.info("Shift images from the points")

    p1, p2 = get_points(shift)
    image, translation = shift_image(p1, p2, image)

    return image, translation


# def get_overlap_between_neighbors(shift1: List[np.ndarray], shift2: List[np.ndarray]) -> np.ndarray:
#     """
#     Get the overlap area between 2 nighbor images. The area of the overlap is the
#     region of the array that has the value 2

#     :param shift1: the shift of the first image
#     :type shift1: List[np.ndarray]
#     :param shift2: the shift of the second image
#     :type shift2: List[np.ndarray]
#     :return: the overlap region
#     :rtype: np.ndarray
#     """

#     # recover the shift from the images
#     image1, _ = shift_image(shift1[0], shift1[1])
#     image2, _ = shift_image(shift2[0], shift2[1])

#     if image1.shape[0]-image2.shape[0] > 0:
#         image2 = np.pad(image2, ((0, image1.shape[0]-image2.shape[0]), (0, 0)),
#                         "constant")
#     else:
#         image1 = np.pad(image1, ((0, image2.shape[0]-image1.shape[0]), (0, 0)),
#                         "constant")
#     if image1.shape[1]-image2.shape[1] > 0:
#         image2 = np.pad(image2, ((0, 0), (0, image1.shape[1]-image2.shape[1])),
#                         "constant")
#     else:
#         image1 = np.pad(image1, ((0, 0), (0, image2.shape[1]-image1.shape[1])),
#                         "constant")

#     # get the overlap between images by adding the 2 big images
#     both_images = image1+image2
#     print('both_images:', both_images)
#     return both_images


# def get_overlap_from_shifts(shifts: Dict[str, np.array],
#                             visualize: bool = False,
#                             out_path: str = None) -> Dict[str, Set[Tuple[int]]]:
#TODO: IF YOU NEED THIS FUNCTION ADAPT YOUR CODE TO USE AFFINETRANSFORM.compute_overlap_region
#     """
#     Find the overlap regions from a montage image where the single images are
#     overlapping because they are taken each degree and are 1.5 degree size

#     :param shifts: the dictionnary containing the shifts from the images
#     :return: the overlapping indices for each image

#     :param shifts: the dictionnary containing the shifts from the images
#     :type shifts: Dict[str, np.array]
#     :param visualize: whether to visualize the overlap or not, defaults to False
#     :type visualize: bool, optional
#     :param out_path: whether the path to save visualized images, defaults to None
#     :type visualize: str, optional
#     :return: the overlapping indices for each image
#     :rtype: Dict[str, Set[Tuple[int]]]
#     """

#     # save the overlapping regions and the images done for overlapping in order
#     # to avoid duplicating the computations
#     overlap_indices = set()
#     overlap_done = []
#     if visualize:
#         all_overlap_image = np.zeros((max(np.max(shifts["bottom_right_y"]),
#                                           np.max(shifts["bottom_left_y"]))-min(np.min(shifts["top_right_y"]),
#                                                                                np.min(shifts["top_left_y"])),
#                                           max(np.max(shifts["bottom_right_x"]),
#                                               np.max(shifts["top_right_x"]))-min(np.min(shifts["top_left_x"]),
#                                                                                  np.min(shifts["bottom_left_x"])), 3))
#     for index, row in shifts.iterrows():
#         image_number = re.search(r"_(\d+)_", row['name']).group(0)
#         location_present = re.search(
#             r'\((.*?)\)', row['name']).group(1).split(',')
#         location_present = [int(location_present[0]), int(location_present[1])]
#         # look for neighbors if the image exists to search for overlap
#         neighbors = [(location_present[0], location_present[1]),
#                      (location_present[0]+1, location_present[1]),
#                      (location_present[0]-1, location_present[1]),
#                      (location_present[0], location_present[1]+1),
#                      (location_present[0], location_present[1]-1),
#                      (location_present[0]-1, location_present[1]-1),
#                      (location_present[0]-1, location_present[1]+1),
#                      (location_present[0]+1, location_present[1]-1),
#                      (location_present[0]+1, location_present[1]+1)]
#         image_neighbors = []
#         session = re.search(r'Session(\d+)', row['name']).group(1)
#         for neighbor in neighbors:
#             neig = [name for name in list(shifts['name']) if str(neighbor).replace(" ", "")
#                     in name and "Session" + session in name]
#             if neig:
#                 # neighbor exists so look for overlapping region between them
#                 neig_number = re.search(r"_(\d+)_", neig[0]).group(0)
#                 if neig_number == image_number:
#                     continue
#                 if not neig_number + image_number in overlap_done:
#                     overlap_done.append(image_number + neig_number)
#                     image = np.ones((IMAGE_SIZE, IMAGE_SIZE))
#                     neig_shift = shifts[shifts['name'] == neig[0]]
#                     image_shift, _ = shift_image([row['top_left_y'], row['top_left_x']],
#                                                  [row['top_right_y'], row['top_right_x']])
#                     image_neig_shift, _ = shift_image([neig_shift['top_left_y'].values[0], neig_shift['top_left_x'].values[0]],
#                                                       [neig_shift['top_right_y'].values[0], neig_shift['top_right_x'].values[0]])
#                     image_shift = np.pad(image_shift, ((0, max(image_shift.shape[0], image_neig_shift.shape[0])-image_shift.shape[0]),
#                                                        (0, max(image_shift.shape[1], image_neig_shift.shape[1])-image_shift.shape[1])), 'constant')
#                     image_neig_shift = np.pad(image_neig_shift, ((0, max(image_shift.shape[0], image_neig_shift.shape[0])-image_neig_shift.shape[0]),
#                                                                  (0, max(image_shift.shape[1], image_neig_shift.shape[1])-image_neig_shift.shape[1])), 'constant')

#                     if visualize:
#                         tot_image = np.array((image_shift, np.zeros(
#                             image_shift.shape), image_neig_shift)).transpose(1, 2, 0)
#                         tot_image = (255*tot_image).astype(np.uint8)
#                         cv2.imwrite(os.path.join(out_path, "first_test.tif"),
#                                     tot_image)
#                         tot_image = np.ascontiguousarray(tot_image)
#                     # get the indices of the overlap for the basic image and the
#                     # neighbor image
#                     image_shift_indices = set(
#                         (tuple(i) for i in np.argwhere(image_shift)))
#                     image_neig_shift_indices = set(
#                         (tuple(i) for i in np.argwhere(image_neig_shift)))
#                     # add to the output dictionnary
#                     overlap_ = image_shift_indices.intersection(
#                         image_neig_shift_indices)
#                     overlap_indices = overlap_indices | overlap_
#                     if visualize:
#                         # draw the overlapping squares
#                         overlap = np.array(list(overlap_))
#                         top = overlap[np.argmin(overlap[:, 0])]
#                         bottom = overlap[np.argmax(overlap[:, 0])]
#                         left = overlap[np.argmin(overlap[:, 1])]
#                         right = overlap[np.argmax(overlap[:, 1])]
#                         tot_image = cv2.rectangle(tot_image, (top[1], top[0]),
#                                                   (bottom[1], bottom[0]),
#                                                   color=(255, 255, 255), thickness=-1)
#                         tot_image = cv2.rectangle(tot_image, (left[1], left[0]),
#                                                   (right[1], right[0]),
#                                                   color=(255, 255, 255), thickness=-1)
#                         try:
#                             all_overlap_image[0:tot_image.shape[0],
#                                               0:tot_image.shape[1], :] += tot_image
#                         except ValueError:
#                             tot_image = tot_image[0:min(tot_image.shape[0], all_overlap_image.shape[0]),
#                                                   0:min(tot_image.shape[1], all_overlap_image.shape[1])]
#                             all_overlap_image[0:tot_image.shape[0],
#                                               0:tot_image.shape[1], :] += tot_image
#                         cv2.imwrite(os.path.join(out_path, "second_test.tif"),
#                                     all_overlap_image.astype(np.uint8))
#     if visualize:
#         cv2.imwrite(os.path.join(out_path, "final_test.tif"), all_overlap_image)
#     return overlap_indices


# def subject_neighbor(filenames: List[str], shifts: pd.DataFrame, distance: bool = True) -> Dict[str, List[str]]:
#     """
#     From a montage image return the difference in distance between 2
#     neighbooring image

#     :param filenames: the name of the files of the subject
#     :type filenames: List[str]
#     :param shifts: the shifts from the montage as a DataFrame with image name
#                    and the points of the image
#     :type shifts: pd.DataFrame
#     :param distance: whether or not we need the distance between neighbors,
#                      defaults to True, defaults to True
#     :type distance: bool, optional
#     :return: the differences in distance between neighbor images
#     :rtype: Dict[str, List[str]]
#     """

#     logging.info("Look for neighbors")
#     # will save the neighbor images from each image in the dictionnary
#     neighbors_image = {}

#     # go through the shifts to find if they are neighbors
#     for _, row in shifts.iterrows():
#         image_shift = row['name']
#         # the new way of writing shift is a list of points, base_points and shifts,
#         # so we need to take this into account and take only the shift

#         # now look for possible neighbors in a squared neighborhood
#         try:
#             location_present = re.search(
#                 r'\((.*?)\)', image_shift).group(1).split(',')
#         except AttributeError:
#             continue
#         location_present = [int(location_present[0]), int(location_present[1])]
#         neighbors = [(location_present[0]+1, location_present[1]),
#                      (location_present[0]-1, location_present[1]),
#                      (location_present[0], location_present[1]+1),
#                      (location_present[0], location_present[1]-1),
#                      (location_present[0]-1, location_present[1]-1),
#                      (location_present[0]-1, location_present[1]+1),
#                      (location_present[0]+1, location_present[1]-1),
#                      (location_present[0]+1, location_present[1]+1)]

#         # will save all the nieghbors from an image from a same session in this list
#         image_neighbors = []
#         session = re.search(r'Session(\d+)', image_shift).group(1)
#         for neighbor in neighbors:
#             neig = [name for name in filenames if
#                     str(neighbor).replace(" ", "") in name and "Session" + session in name]
#             if neig:
#                 # the image has an other image as neighbor
#                 calculated_split_neig = re.sub(
#                     "Confocal", "CalculatedSplit", neig[0])
#                 if distance:
#                     # if you want to have the distance as output as well, search for it
#                     # has the form dx ->, dy up and image_name is of the form dx ->, dy down
#                     diff_location = [-1 if location_present[0] < 0 or location_present[1] < 0
#                                      else 1][0]
#                     # shift: x to the right, y down [x,y]
#                     neig_shift = shifts[shifts['name'] == neig[0]]
#                     image_shift_ = shifts[shifts['name'] == image_shift]
#                     dist_between_neighbors_mat = (neig_shift['top_left_y'].values[0] -
#                                                   image_shift_['top_left_y'].values[0],
#                                                   neig_shift['top_left_x'].values[0] -
#                                                   image_shift_['top_left_x'].values[0])
#                     dist_between_neighbors_mat = [diff_location*dist_between_neighbors_mat[i]
#                                                   for i in range(len(dist_between_neighbors_mat))]

#                     # Check whether the images are overlapping, if not continue
#                     if np.abs(dist_between_neighbors_mat[0]) > IMAGE_SIZE or \
#                             np.abs(dist_between_neighbors_mat[1]) > IMAGE_SIZE:
#                         continue
#                     # we found a neighbor so we add it to the list
#                     image_neighbors.append(
#                         [calculated_split_neig, dist_between_neighbors_mat])
#                 else:
#                     # no distance needed so just append the neighbor to the list
#                     image_neighbors.append([calculated_split_neig])
#         # add the list to the dictionnary whil also adding the basic image as the
#         # closest neighbor
#         calculated_split = re.sub("Confocal", "CalculatedSplit", image_shift)
#         if distance:
#             image_neighbors.append([calculated_split, (0, 0)])
#         neighbors_image[calculated_split] = image_neighbors

#     return neighbors_image


def find_smaller(image1: str, image2: str) -> str:
    """
    Find which of the image is the smallest between 2 image name and returns it

    :param image1: the first image name
    :type image1: str
    :param image2: the second image name
    :type image2: str
    :return: the smallest image name between the 2
    :rtype: str
    """
    location1 = eval(re.search(r"\((.*?)\)", image1).group(0))
    location2 = eval(re.search(r"\((.*?)\)", image2).group(0))
    if location1[0] >= location2[0] and location1[1] >= location2[1]:
        return image1
    else:
        return image2


def should_be_increased(location1: Tuple[int, int], shift1: List[np.ndarray],
                        shift2: List[np.ndarray]) -> bool:
    """
    Look whether the image from the shift2 needs to be increase
    [[1,0,dy][0,1,dx]], (y,x), location (x,y)

    :param location1: the location of the first image
    :type location1: Tuple[int, int]
    :param shift1: the shift of the first image
    :type shift1: List[np.ndarray]
    :param shift2: the shift of the second image
    :type shift2: List[np.ndarray]
    :return: True if the image needs to be shifted otherwise False
    :rtype: bool
    """

    if location1[0] == 0:
        # shift in y is important
        try:
            if shift2['top_left_y'].values[0] > shift1['top_left_y'].values[0]:
                return True
        except AttributeError:
            if shift2['top_left_y'] > shift1['top_left_y'].values[0]:
                return True
    elif location1[1] == 0:
        # shift in x is important -> bigger shift in x means need to move
        try:
            if shift2['top_left_x'].values[0] > shift1['top_left_x'].values[0]:
                return True
        except AttributeError:
            if shift2['top_left_x'] > shift1['top_left_x'].values[0]:
                return True

    return False


def increase_next_neighbors(image1: str, shifts: Dict[str, List[np.ndarray]],
                            indice: Tuple[int, int], center: Tuple[int, int],
                            center_loc: Tuple[int, int]
                            ) -> Tuple[Dict[str, List[np.ndarray]], Tuple[int, int]]:
    """
    Look for all neighbors that need to be added some shifts. The selected image
    to be shift are the one on the right or on bottom of the image

    :param image1: the image name where the SSIM was found
    :type image1: str
    :param shifts: the dictionnary of shifts
    :type shifts: Dict[str, List[np.ndarray]]
    :param indice: the indice found to be the better correction
    :type indice: Tuple[int, int]
    :param center: the center pixel location
    :type center: Tuple[int, int]
    :param center_loc: in which image the center is
    :type center_loc: Tuple[int, int]
    :return: the upodated shifts
    :rtype: Tuple[Dict[str, List[np.ndarray]], Tuple[int, int]]
    """

    shift_image = shifts[shifts['name'] == image1]
    location1 = eval(
        re.search(r"\((.*?)\)", shift_image['name'].values[0]).group(0))

    # look if we need to increase the center position
    if center_loc[0] <= location1[0] and center_loc[1] >= location1[1] and not \
            location1 == center_loc:
        center = [center[0] + indice[1], center[1] + indice[0]]

    for i, shift in shifts.iterrows():
        if shift['name'] == image1:
            continue

        # look if the image needs to be increase and increase if so
        if should_be_increased(location1, shift_image, shift):
            shift['top_left_y'] += indice[0]
            shift['bottom_left_y'] += indice[0]
            shift['top_right_y'] += indice[0]
            shift['bottom_right_y'] += indice[0]
            shift['top_left_x'] += indice[1]
            shift['bottom_left_x'] += indice[1]
            shift['top_right_x'] += indice[1]
            shift['bottom_right_x'] += indice[1]
            shifts.at[i] = shift

    return shifts, center


def get_neighbor_patch_distance(patches_dir: str,
                                neighbors_image: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Get the nighbooring patch distance of a specific patch to look which one to
    take in the NST process

    :param patches_dir: the directory with all the patches image that can be neighbor
    :type patches_dir: str
    :param neighbors_image: the image to find neighbors to
    :type neighbors_image: Dict[str, List[str]]
    :return: the neighbors with their distance to the image we want to find neighbor
    :rtype: Dict[str, float]
    """

    # Now we can use neighbors_image to find the references patches around a certain location
    # Take patches where relative position < 480 pixels
    # replace os.path.join(base_dir, "first_run") by output_dir
    all_patches = os.listdir(os.path.join(patches_dir, RAW_DATA_DIR_NAME))
    neig_patch_dic = {}
    for patch_name in all_patches:
        closest_patches = []
        if "Thumbs" in patch_name: continue
        xy_position = (re.search(r"_x(\d+)y(\d+)", patch_name).group(1),
                       re.search(r"_x(\d+)y(\d+)", patch_name).group(2))
        image_name = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", patch_name)
        if image_name in neighbors_image.keys():
            neighbor_patches = []
            for neighbor in neighbors_image[image_name]:
                neig_patches = [[patch, neighbor[1]]
                                for patch in all_patches if neighbor[0] in patch]
                neighbor_patches.extend(neig_patches)
            for neighbor_patch in neighbor_patches:
                neighbor_patch_xy_position = (re.search(r"_x(\d+)y(\d+)", neighbor_patch[0]).group(
                    1), re.search(r"_x(\d+)y(\d+)", neighbor_patch[0]).group(2))

                # Euclidean distance between the patches
                dist = np.sqrt((int(neighbor_patch_xy_position[0])-int(xy_position[0])+neighbor_patch[1][0])**2 +
                               (int(neighbor_patch_xy_position[1])-int(xy_position[1])+neighbor_patch[1][1]) ** 2)
                if 0 < dist < 480:
                    closest_patches.append([neighbor_patch[0], dist])
        neig_patch_dic[patch_name] = closest_patches

    return neig_patch_dic


def inverse_x_axis(densities: Dict[int, float]) -> Dict[int, float]:
    """
    Inverse the density values for the x axis because it was inversed in the
    AOSLO machine before Subject66 and for some subjects that have been taken
    the left eye instead of the right one

    :param densities: the density to inverse
    :type densities: Dict[int, float]
    :return: the inversed density
    :rtype: Dict[int, float]
    """

    reversed_density = {}
    for key, val in densities.items():
        reversed_density[-key] = val

    return reversed_density


def get_center_peak(layer: np.ndarray) -> Tuple[int, int]:
    """
    Get the center oeak of a 3d surface

    :param layer: the 3d surface
    :type layer: np.ndarray
    :return: the peak of the layer
    :rtype: Tuple[int, int]
    """

    peak_layers = []
    peaks = []
    for single_layer in layer:
        # only works with 1d arrays
        peak_layers.append(find_peaks(
            single_layer, distance=1729, prominence=0.1, width=5))
    for peak in peak_layers:
        try:
            new_peak = find_peaks(
                np.array(layer)[:, peak[0][0]], distance=97, prominence=0.1, width=5)
            try:
                new_peak[0][0]
            except IndexError:
                peaks.append(0)
                continue
            peaks.append(new_peak)
        except IndexError:
            peaks.append(0)

    # Gaussian Filter the layer for the plots
    layer = np.array(layer)
    layer_5 = gaussian_filter(layer, sigma=5)
    pos_layer_5 = layer_5[round(layer_5.shape[0]/3):-round(
        layer_5.shape[0]/3), round(layer_5.shape[1]/3):-round(layer_5.shape[1]/3)]
    neg_layer_5 = layer_5[round(layer_5.shape[0]/3):-round(
        layer_5.shape[0]/3), round(layer_5.shape[1]/3):-round(layer_5.shape[1]/3)]
    try:
        pos_peak = np.unravel_index(
            pos_layer_5.argmax(), pos_layer_5.shape)
    except ValueError:
        raise

    peak = (pos_peak[0]+round(layer.shape[0]/3),
            pos_peak[1]+round(layer.shape[1]/3))

    return peak


def get_otsu_threshold(cones_detected_hist: Tuple[np.array, np.array]) -> float:
    """
    Get the OTSU threshold of an histogram

    :param cones_detected_hist: histogram of the number of cones detected
    :type cones_detected_hist: Tuple[np.array, np.array]
    :return: the threshold corresponding to the bins to separate the histogram
    :rtype: float
    """

    # OTSU threshold
    counts = cones_detected_hist[0]
    bin_centers = 0.5*(cones_detected_hist[1][1:]+cones_detected_hist[1][:-1])

    # We can try Otsu (or other) thresholds on cones_detected_hist:
    # counts, bin_centers = _validate_image_histogram(image, hist, nbins)
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # assert len([w for w in np.append(weight1, weight2) if w==0]) == 0, \
    #     "The weights of the histogram are null, please verify your histogram"
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def get_neighbor_pixel(m: np.ndarray, x: int, y: int, circle: bool = False,
                       r: int = MDRNN_OVERLAP_DISTANCE) -> List[List[int]]:
    """
    Search for a positive pixel in the neighborhood of a given pixel

    :param m: the array to look into to find positive pixels
    :type m: np.ndarray
    :param x: the x position of the given pixel
    :type x: int
    :param y: the y position of the given pixel
    :type y: int
    :param circle: true if you want to look for a circular neighborhood else
                   squared neighborhood, defaults to False
    :type circle: bool, optional
    :param r: the radius distance to look for neighbors, defaults to MDRNN_OVERLAP_DISTANCE
    :type r: int, optional
    :return: the list of positions of the neighbors
    :rtype: List[List[int]]
    """

    neighbor = []
    for i_x in range(-r, r+1):
        for i_y in range(-r, r+1):
            new_x = i_x + x
            new_y = i_y + y
            if 0 <= new_x < m.shape[0] and 0 <= new_y < m.shape[1]:
                if m[new_x, new_y]:
                    if circle:
                        if np.sqrt((i_x)**2 + (i_y)**2) <= r:
                            neighbor.append([new_x, new_y])
                    else:
                        if abs(i_x) <= r and abs(i_y) <= r:
                            neighbor.append([new_x, new_y])
    return neighbor


def find_neighbors(m: np.ndarray, i: int, j: int, dist: int = 1) -> np.ndarray:
    """
    Search in a neighborhood distance (dist) of the positions (i,j) if there is
    a positive value in the array and returns the array with one if there is

    :param m: the array with ones corresponding to cones and 0 to no cones
    :type m: np.ndarray
    :param i: the y position
    :type i: int
    :param j: the x position
    :type j: int
    :param dist: the distance to search neighbors, defaults to 1
    :type dist: int, optional
    :return: the local array with ones corresponding to cones and 0 to no cones
    :rtype: np.ndarray
    """

    return np.array([row[max(0, j-dist):min(j+dist+1, m.shape[0])] for row in m[max(0, i-dist):min(i+dist+1, m.shape[0])]])


def get_overlaping_cones(cones_location_image: Dict[str, Dict[str, Tuple[str, str]]],
                         inverse: bool = False,
                         neighbor_dist: int = MDRNN_OVERLAP_DISTANCE_SAME_IMAGE
                         ) -> Dict[str, np.array]:
    """
    Look if there are overlapping cones in images due to patch overlapping
    The coordinates are coming in (y,x) if from Davidson -> inverse = False

    :param cones_location_image: dictionnary with each image, location of the patch
                                 on the image and locations of the cones detected
    :type cones_location_image: Dict[str, Dict[str, Tuple[str, str]]]
    :param inverse: if the cone centers are of the form (x,y) then inverse = True, defaults to False
    :type inverse: bool, optional
    :param neighbor_dist: distance between the cone centers to merge them when the
                          patches are overlapping on the big image, defaults to
                          MDRNN_OVERLAP_DISTANCE_SAME_IMAGE
    :type neighbor_dist: int, optional
    :raises ValueError: if the transformation is out of the image size
    :return: a dictionnary with the number of the image and a binary map with ones
             where there are cones and zero otherwise
    :rtype: Dict[str, np.array]
    """

    cone_image_locations = {}
    for image, value in cones_location_image.items():
        # the location of the patch must be found to find the real position of the
        # cone in the big image. Normally a patch is 185x185 pixels and an image
        # is 720x720 pixels.
        locations = [(eval(elem[0]), eval(elem[1]))
                     for elem in list(value.keys())]
        cones = list(value.values())
        all_locations = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        if locations:
            for i in range(len(locations)):
                for cone in cones[i]:
                    if inverse:
                        new_location = [locations[i][1] + cone[0],
                                        locations[i][0] + cone[1]]
                    else:
                        new_location = [locations[i][0] + cone[0],
                                        locations[i][1] + cone[1]]
                    if new_location[0] > IMAGE_SIZE:
                        raise ValueError('The new cone center is outside of the \
                                          image size, you might want to inverse \
                                          the boolean to reverse the cone location \
                                          (x,y) or increase the total image size.')
                    # sometimes there is a round error so substract 1 pixel when
                    # the cone is on the border of the image
                    if new_location[0] == IMAGE_SIZE:
                        new_location[0] -= 1
                    if new_location[1] == IMAGE_SIZE:
                        new_location[1] -= 1
                    # look if there is a cone in the neighborhood of the cone we
                    # want to add in the big image
                    neighbors = get_neighbor_pixel(all_locations, new_location[0],
                                                   new_location[1], NIEGHBOR_CIRCULAR_SHAPE,
                                                   neighbor_dist)
                    if neighbors:
                        # merge the cone centers if they are neighbors
                        for neighbor in neighbors:
                            all_locations[neighbor[0], neighbor[1]] = 0
                        xs = [new_location[1]] + [neighbor[1]
                                                  for neighbor in neighbors]
                        ys = [new_location[0]] + [neighbor[0]
                                                  for neighbor in neighbors]
                        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
                        all_locations[round(centroid[1]),
                                      round(centroid[0])] = 1
                    else:
                        all_locations[new_location[0], new_location[1]] = 1
            cone_image_locations[image] = all_locations
        else:
            cone_image_locations[image] = []

    return cone_image_locations


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


def save_checkpoint(checkpoint_path: str, *pickle_objects) -> None:
    """
    Save pickle objects to a checkpoint directory for long analysis

    :param checkpoint_path: the path where to save the pickle objects
    :type checkpoint_path: str
    :params pickle_objects: the dictionnaty of the objects to save as pickles
    """
    for pickle_object in pickle_objects:
        with open(os.path.join(checkpoint_path, f"{list(pickle_object.keys())[0]}.pickle"), 'wb') as handle:
            pickle.dump(list(pickle_object.values())[
                        0], handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(checkpoint_path: str, *pickle_objects) -> List:
    """
    Load pickle objects from a checkpoint directory for long analysis

    :param checkpoint_path: the path where to load the pickle objects
    :type checkpoint_path: str
    :params pickle_objects: the objects to load as pickles
    :return: the list of the loaded pickle objects
    :rtype: List
    """
    objects = []
    for pickle_object in pickle_objects:
        try:
            pickle_object_name = os.path.join(
                checkpoint_path, f"{str(pickle_object)}.pickle")
            with open(pickle_object_name, 'rb') as handle:
                pickle_object = pickle.load(handle)
            objects.append(pickle_object)
        except FileNotFoundError:
            logging.error(f"The file {pickle_object_name} cannot be found, \
                            make sure it exists.")

    return objects


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


def read_csv_cones(file: CoordinatesFile, cone_path: str) -> np.array:
    """
    Get the number of cones detected for each patch

    :param file: the file as a csv where the cone center are
    :type file: CoordinatesFile
    :param cone_path: the path where the csv file is
    :type cone_path: str
    :return: an array with the cone locations as (y,x) for OpenCV images
    :rtype: np.array
    """

    csv_file = CoordinatesFile(file.name)
    csv_file.read_data(cone_path)

    return csv_file.data


def get_cone_csv_file_path(output_path: str) -> str:
    """
    Get the path where the csv files from the MDRNN cone detection is

    :param output_path: output path of the cone detection
    :type output_path: str
    :return: the path where the csv files are
    :rtype: str
    """

    cone_dir = MDRNN_CONE_DETECTOR_OUT_DIR
    cone_path = os.path.join(output_path, cone_dir, "algorithmMarkers")

    return cone_path


def cones_location_per_patch(output_path: str,
                             secondary_path: str = None,
                             third_path: str = None) -> Dict[str, Tuple[int, int]]:
    """
    Function to gather the number of cones detected from a csv file

    :param output_path: output directory where the cones are
    :param secondary_path: secondary output directory where the cones are if
                           for example we use a NST directory as the "principal"
                           output we can then correct if some images have really
                           few cones detected (often var without NST)
    :param third_path: third output directory where the cones are if
                           for example we use a NST directory as the "principal"
                           output we can then correct if some images have really
                           few cones detected (often std without NST)
    :return: a dictionnary with the patch number as key and the indices of the
             cones as (y,x) for OpenCV standards

    :param output_path: output directory where the cones are
    :type output_path: str
    :param secondary_path: secondary output directory where the cones are if
                           for example we use a NST directory as the "principal"
                           output we can then correct if some images have really
                           few cones detected (often var without NST), defaults to None
    :type secondary_path: str, optional
    :param third_path: third output directory where the cones are if
                           for example we use a NST directory as the "principal"
                           output we can then correct if some images have really
                           few cones detected (often std without NST), defaults to None
    :type third_path: str, optional
    :return: a dictionnary with the patch number as key and the indices of the
             cones as (y,x) for OpenCV standards
    :rtype: Dict[str, Tuple[int, int]]
    """

    # look for the directory with the cones annotated
    # output dir is the base directory of output -> look for output from Davidson's algorithm
    cone_path = get_cone_csv_file_path(output_path)
    if secondary_path:
        # the var directory without NST
        cone_secondary_path = get_cone_csv_file_path(secondary_path)
    if third_path:
        # the std directory without NST
        cone_third_path = get_cone_csv_file_path(third_path)

    cones_locations = {}

    number_of_entries = len(os.listdir(cone_path))
    logging.info("Reading cone detection output")
    for file in tqdm(os.scandir(cone_path), desc=f"Reading cone detection output, {number_of_entries=}"):
        best_path = cone_path
        number_first_cones = len(read_csv_cones(file, cone_path))
        best_number = number_first_cones
        if secondary_path:
            # Take the best of all the cones
            number_second_cones = len(read_csv_cones(file, cone_secondary_path))
            if number_second_cones > best_number:
                best_path = cone_secondary_path
                best_number = number_second_cones
        if third_path:
            # Take the best of all the cones
            number_third_cones = len(read_csv_cones(file, cone_third_path))
            if number_third_cones > best_number:
                best_path = cone_third_path
                best_number = number_third_cones
        cones_locations[os.path.splitext(
            file.name)[0]] = read_csv_cones(file, best_path)

    return cones_locations


def number_of_cones_per_patch(output_path: str) -> Dict[str, int]:
    """
    Function to get the number of cones detected from a csv file of cone positions

    :param output_path: output directory where the cones are

    :param output_path: output directory where the cones are
    :type output_path: str
    :return: a dictionary with the patches and their number of cones detected
    :rtype: Dict[str, int]
    """

    # look for the directory with the cones annotated
    # output dir is the base directory of output -> look for output from Davidson's algorithm
    cone_path = get_cone_csv_file_path(output_path)
    number_cones_detected = {}

    number_of_entries = len(os.listdir(cone_path))
    # logging.info("Reading cone detection output")
    for file in tqdm(os.scandir(cone_path),
                     desc=f"Reading cone detection output, {number_of_entries=}"):
        # count the number of cones detected for each patch
        number_cones_detected[os.path.splitext(file.name)[0]] = \
            len(read_csv_cones(file, cone_path))
    return number_cones_detected


def cones_location_per_image(cones_location_patch: Dict[str, Tuple[int, int]]
                             ) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """
    Reassemble cone locations from patches into a dictionnary with locations

    :param cones_location_patch: cone locations in each patch
    :type cones_location_patch: Dict[str, Tuple[int, int]]
    :return: the reassembled dictionnary for each images
    :rtype: Dict[str, Dict[str, Tuple[str, str]]]
    """

    cones_location_image = {}
    for key, val in cones_location_patch.items():
        # find the patch location from its prefix
        crop_prefix = re.search(r'CROP_((\d+))_x((\d+))y((\d+))_', key)
        # positions has the form (x,y)
        crop_positions = (crop_prefix[3], crop_prefix[5])
        name = re.sub(crop_prefix[0], "", key)
        number = re.search(r'_(\d+)_', name)[0]
        if number in cones_location_image.keys():
            cones_location_image[number][crop_positions] = val
        else:
            cones_location_image[number] = {}
            cones_location_image[number][crop_positions] = val

    return cones_location_image


def get_cones_mdrnn(output_path: str,
                    secondary_path: str = None,
                    third_path: str = None) -> Dict[str, np.array]:
    """
    Get cones on the images based on cone detected by Davidson's MDRNN

    :param output_path: output directory where the cones are
    :type output_path: str
    :param secondary_path: secondary output directory where the cones are if
                           for example we use a NST directory as the "principal"
                           output we can then correct if some images have really
                           few cones detected (often var without NST), defaults to None
    :type secondary_path: str, optional
    :param third_path: third output directory where the cones are if
                        for example we use a NST directory as the "principal"
                        output we can then correct if some images have really
                        few cones detected (often std without NST), defaults to None
    :type third_path: str, optional
    :return: the cone locations for each images as binary map with ones for cone
             centers and zeros otherwise
    :rtype: Dict[str, np.array]
    """

    # get locations per patch and then per images
    cones_location_patch = cones_location_per_patch(output_path, secondary_path,
                                                    third_path)
    cones_location_image = cones_location_per_image(cones_location_patch)

    # Now look if there are overlapping patches in order to remove them
    try:
        # cones have coordinates (y,x) and patch locations (y,x) so inverse = True
        cones_tot_location_image = get_overlaping_cones(cones_location_image)
    except ValueError as error:
        logging.exception(error)

    return cones_tot_location_image

def get_cones_atms(output_path: str) -> Dict[str, np.array]:
    """
    Get cones on the images based on cone detected by ATMS

    :param output_path: output directory where the cones are
    :type output_path: str
    :return: the cone locations for each images as binary map with ones for cone
             centers and zeros otherwise
    :rtype: Dict[str, np.array]
    """

    # read the files containing the cone centers
    cones_per_patch = {}
    number_of_entries = len(os.listdir(output_path))
    # logging.info("Reading cone detection output")
    for file in tqdm(os.scandir(output_path),
                     desc=f"Reading cone detection output, {number_of_entries=}"):
        if file.name.endswith(".csv"):
            cones_per_patch[os.path.splitext(
                file.name)[0]] = read_csv_cones(file, output_path)
    # reconstruct the images with cones
    cones_per_image = cones_location_per_image(cones_per_patch)
    try:
        # cones have coordinates (y,x) but patch locations (x,y) so inverse = True
        cones_per_image = get_overlaping_cones(cones_per_image, inverse=True,
                                               neighbor_dist=0)
    except ValueError as error:
        logging.exception(error)

    return cones_per_image


def get_overlap(montaged_dir: str) -> Dict[str, np.array]:
    """
    Get the overlapping regions from the montaged corrected directory

    :param montaged_dir: the montage directory where the files ares
    :type montaged_dir: str
    :return: the dictionnary with the image files and their overlap
    :rtype: Dict[str, np.array]
    """

    # construct the Data Structure to store outputs
    overlap_regions = {}
    already_overlapped = []
    # change to parser.get.montaged_dir
    locations = {}

    # first get from each image an array with the locations where the image pixel
    # value is not zero
    for image_name in os.listdir(montaged_dir):
        image_dir = os.path.join(montaged_dir, image_name)
        if not os.path.isdir(image_dir):
            if "Subject" in image_name and "Confocal" in image_name:
                image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
                filled_pixels = np.argwhere(image > 0)
                locations[image_name] = filled_pixels

    # then look for existing neighbors for each image
    for image_name, region in locations.items():
        location = re.search(r'\((.*?)\)', image_name).group(1).split(',')
        location_present = [int(location[0]), int(location[1])]
        neighbors = [(location_present[0]+1, location_present[1]),
                     (location_present[0]-1, location_present[1]),
                     (location_present[0], location_present[1]+1),
                     (location_present[0], location_present[1]-1),
                     (location_present[0]-1, location_present[1]-1),
                     (location_present[0]-1, location_present[1]+1),
                     (location_present[0]+1, location_present[1]-1),
                     (location_present[0]+1, location_present[1]+1)]
        image_neighbors = []
        top = np.min(region[:, 0])
        left = np.min(region[:, 1])
        for neighbor in neighbors:
            neig = [name for name in list(locations.keys()) if str(
                neighbor).replace(" ", "") in name]
            if neig:
                # find the area that overlaps between neighboor images
                if {neig[0], image_name} not in already_overlapped:
                    neig_region = locations[neig[0]]
                    df1 = pd.DataFrame(
                        {"x": region[:, 0], "y": region[:, 1]}).reset_index()
                    df2 = pd.DataFrame(
                        {"x": neig_region[:, 0], "y": neig_region[:, 1]}).reset_index()
                    result = pd.merge(df1, df2, left_on=[
                                      "x", "y"], right_on=["x", "y"])

                    out_name = re.sub(
                        r"_aligned_to_ref(\d+)_m(\d+)", "", image_name)
                    if out_name in overlap_regions.keys():
                        overlap_regions[out_name] = np.concatenate(
                            (overlap_regions[out_name], result[["x", "y"]].to_numpy() - np.array((top, left))))
                    else:
                        overlap_regions[out_name] = result[[
                            "x", "y"]].to_numpy() - np.array((top, left))

                    already_overlapped.append({image_name, neig[0]})

    return overlap_regions


def pixels_per_patch(patch_dir: str) -> Dict[str, np.array]:
    """
    Function to get the pixel values of every patches used

    :param patch_dir: The patch directory
    :type patch_dir: str
    :return: a dictionary with all the pixel values that are in the patch
    :rtype: Dict[str, np.array]
    """
    # look for the patches used to run Davidson's algorithm
    # output dir is the base directory of output -> look for output from
    # Davidson's algorithm

    pixels = {}

    number_of_entries = len(os.listdir(patch_dir))
    # logging.info("Reading cone detection output")
    for file in tqdm(os.scandir(patch_dir),
                     desc=f"Reading pixels in patch images, {number_of_entries=}"):
        if os.path.isfile(os.path.join(patch_dir, file)):
            if file.name.endswith(".tif"):
                # get histogram of single patch image
                image = ImageFile(file.name)
                image.read_data(patch_dir)
                hist = np.histogram(image.data, 256, [0, 256])
                pixels[file.name] = hist

    return pixels


def get_transform_from_shifts(shift: pd.DataFrame) -> np.ndarray:
    """
    Get the transform matrix from the points in the shift DataFrame

    :param shift: the dhits of the images
    :type shift: pd.DataFrame
    :return: their affine transform
    :rtype: np.ndarray
    """

    base_points = np.float32([[0, 0], [719, 0], [0, 719]])  # , [719, 719]])
    points = np.float32([[shift['top_left_y'], shift['top_left_x']],
                         [shift['bottom_left_y'], shift['bottom_left_x']],
                         [shift['top_right_y'], shift['top_right_x']]])
    M = np.array(cv2.getAffineTransform(base_points, points))
    dx = M[0, 2]
    M[0, 2] = M[1, 2]
    M[1, 2] = dx

    return M


def get_transform(image_padded: np.ndarray, image_normal: np.ndarray) -> np.ndarray:
    """
    Get the transformation between a padded image and an other image

    :param image_padded: the zero-padded image
    :type image_padded: np.ndarray
    :param image_normal: the other image to find the transform to
    :type image_normal: np.ndarray
    :return: the affine transform for translation of the form [[1,0,dy], [0,1,dx], [0,0,1]]
    :rtype: np.ndarray
    """

    top, bottom, left, right = get_boundaries(image_padded)

    image_points = np.float32(
        [[top, left], [bottom, left], [top, right], [bottom, right]])
    base_points = np.float32([[0, 0], [image_normal.shape[0]-1, 0], [0, image_normal.shape[1]-1],
                              [image_normal.shape[0]-1, image_normal.shape[1]-1]])

    M = cv2.getPerspectiveTransform(base_points, image_points)

    return np.array(M)


def cKDTree_nn(data: List[List[int]]) -> List[int]:
    """
    Computes the Nearest Neighbor distance between points

    :param data: the datapoints corresponding to cone centers
    :type data: List[List[int]]
    :return: nearest neighbor distance
    :rtype: List[int]
    """

    tree = cKDTree(data)
    dists = tree.query(data, 2)
    nn_dist = dists[0][:, 1]

    return nn_dist
