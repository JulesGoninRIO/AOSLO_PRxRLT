import numpy as np
import os
import pickle
import logging
import cv2
import math
import matplotlib.pyplot as plt
from scipy import io
from skimage.metrics import structural_similarity as ssim
from typing import List, Tuple

from src.cell.montage.montage_element import MontageElement
from src.shared.numpy.arrays import get_boundaries
from src.shared.numpy.arrays import mask_array, get_filled_pixels_corners, pad_area_both_directions, adjust_array
from src.shared.helpers.intervals import adjust_intervals

def compute_ssim(img1: np.ndarray, img2: np.ndarray, overlap_region: List[int]) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    This method crops the input images to the specified overlap region and computes the SSIM score between them.

    :param img1: The first image as a numpy array.
    :type img1: np.ndarray
    :param img2: The second image as a numpy array.
    :type img2: np.ndarray
    :param overlap_region: The region of overlap to consider for SSIM computation, specified as [x1, y1, x2, y2].
    :type overlap_region: List[int]
    :return: The SSIM score between the cropped regions of the two images.
    :rtype: float
    """
    # Crop the images to the overlap region
    img1_cropped = img1[overlap_region[0]:overlap_region[2], overlap_region[1]:overlap_region[3]]
    img2_cropped = img2[overlap_region[0]:overlap_region[2], overlap_region[1]:overlap_region[3]]

    # Compute the SSIM score
    ssim_score = ssim(img1_cropped, img2_cropped)

    return ssim_score

def find_best_ssim(img1: np.ndarray, img2: np.ndarray, overlap_region: List[int]) -> Tuple[float, Tuple[int, int]]:
    """
    Find the best SSIM score by shifting the second image.

    This method tries shifting the second image in each direction within the overlap region and computes the SSIM score.
    It returns the best SSIM score and the corresponding shift.

    :param img1: The first image as a numpy array.
    :type img1: np.ndarray
    :param img2: The second image as a numpy array.
    :type img2: np.ndarray
    :param overlap_region: The region of overlap to consider for SSIM computation, specified as [x1, y1, x2, y2].
    :type overlap_region: List[int]
    :return: A tuple containing the best SSIM score and the corresponding shift (dx, dy).
    :rtype: Tuple[float, Tuple[int, int]]
    """
    best_ssim = -np.inf
    best_shift = (0, 0)

    # Try shifting the image in each direction
    for dx in range(-overlap_region[0], overlap_region[2]):
        for dy in range(-overlap_region[1], overlap_region[3]):
            # Shift the image
            img2_shifted = np.roll(img2, shift=(dx, dy), axis=(0, 1))

            # Compute the SSIM score
            ssim_score = compute_ssim(img1, img2_shifted, overlap_region)

            # If the SSIM score is better, update the best SSIM score and the best shift
            if ssim_score > best_ssim:
                best_ssim = ssim_score
                best_shift = (dx, dy)

    return best_ssim, best_shift

class SSIM():
    """
    Class that handles the brut-force SSIM search in two neighborhood images
    without parallelization
    """

    def __init__(
        self,
        element1: MontageElement,
        element2: MontageElement,
        ssim_path: str,
        n: int = 10) -> None:
        """
        SSIM class creator to build the images, mask and the number of pixel
        to search around the overlap region
        """
        self.element1 = element1
        self.element2 = element2
        self.n = n

        # output matrice with ssim scores m
        # Let's first build an empty matrice where we will then store the results
        self.m = np.zeros((2*self.n+1, 2*self.n+1))
        self.ssim_path = ssim_path
        # try:
        # overlap_region = element1.transform.compute_overlap_region(element2.transform)
        self.mask = np.logical_and(element1.transformed_mask != 0, element2.transformed_mask != 0)

        self.img1_copy = self.element1.transformed_data.copy()
        self.img2_copy = self.element2.transformed_data.copy()
        # except ValueError:
        #     mask1 = element1.transformed_mask
        #     mask2 = element2.transformed_mask

        #     # Determine the size of both arrays
        #     size1 = mask1.shape
        #     size2 = mask2.shape

        #     # Pad the smaller array with zeros to match the size of the larger array
        #     if size1 != size2:
        #         max_size = (max(size1[0], size2[0]), max(size1[1], size2[1]))
        #         padded_mask1 = np.zeros(max_size, dtype=mask1.dtype)
        #         padded_mask2 = np.zeros(max_size, dtype=mask2.dtype)
        #         padded_mask1[:size1[0], :size1[1]] = mask1
        #         padded_mask2[:size2[0], :size2[1]] = mask2
        #     else:
        #         padded_mask1 = mask1
        #         padded_mask2 = mask2
        #     self.mask = np.logical_and(padded_mask1 != 0, padded_mask2 != 0)

    def run(self) -> None:
        """
        Function to get a matrice of the SSIM score around a neighborhood of n pixels
        without any parallelization
        """
        # Look for the SSIM score around the indices
        for i in range(-self.n, self.n+1):
            for j in range(-self.n, self.n+1):
                self.run_ssim(i, j)

    def run_ssim(self, i: int, j: int) -> int:
        """
        Run the SSIM function for parallel processing

        :param i: the y direction pixel shfit
        :type i: int
        :param j: the x direction pixel shfit
        :type j: int
        :return: the result of SSIM
        :rtype: int
        """

        # cut the mask and the images so that we only have the region of
        # interest for this pixel alignement (i,j)

        resized_mask = self.resize_mask(i, j)
        img1_cut = mask_array(self.img1_copy, self.mask)
        img2_cut = mask_array(self.img2_copy, resized_mask)

        filled_pixels1 = get_filled_pixels_corners(self.mask)
        filled_pixels2 = get_filled_pixels_corners(resized_mask)
        if filled_pixels1 is None or filled_pixels2 is None:
            # too small array
            self.m[i, j] = 0
            return 0

        top1, bottom1, left1, right1 = filled_pixels1
        top2, bottom2, left2, right2 = filled_pixels2

        img1_cut, img2_cut, top1, bottom1, left1, right1, top2, bottom2, left2, right2 = self.adjust_dimensions(
            img1_cut, img2_cut, top1, bottom1, left1, right1, top2, bottom2, left2, right2)

        if top2 < 0:
            img2_cut = np.pad(img2_cut, ((-top2, 0), (0, 0)), 'constant')
        if left2 < 0:
            img2_cut = np.pad(img2_cut, ((0, 0), (-left2, 0)), 'constant')
        if bottom2 > img2_cut.shape[0]:
            img2_cut = np.pad(img2_cut, ((0, bottom2 - img2_cut.shape[0]), (0, 0)), 'constant')
        if right2 > img2_cut.shape[1]:
            img2_cut = np.pad(img2_cut, ((0, 0), (0, right2 - img2_cut.shape[1])), 'constant')

        img1_cut = img1_cut[top1:bottom1, left1:right1]
        img2_cut = img2_cut[top2:bottom2, left2:right2]

        ratio = self.calculate_ratio(img1_cut, img2_cut)
        if ratio == 0:
            self.m[i, j] = 0
            return 0

        ssim_score = ratio * ssim(img1_cut, img2_cut)
        self.m[i + self.n, j + self.n] = ssim_score
        return ssim_score

    def adjust_dimensions(
        self,
        img1_cut: np.ndarray,
        img2_cut: np.ndarray,
        top1: int,
        bottom1: int,
        left1: int,
        right1: int,
        top2: int,
        bottom2: int,
        left2: int,
        right2: int) -> Tuple[np.ndarray, np.ndarray, int, int, int, int, int, int, int, int]:
        """
        Adjust the dimensions of two image cuts to make them compatible.

        This method adjusts the dimensions of two image cuts by padding them to ensure they have the same size.

        :param img1_cut: The first image cut as a numpy array.
        :type img1_cut: np.ndarray
        :param img2_cut: The second image cut as a numpy array.
        :type img2_cut: np.ndarray
        :param top1: The top boundary of the first image cut.
        :type top1: int
        :param bottom1: The bottom boundary of the first image cut.
        :type bottom1: int
        :param left1: The left boundary of the first image cut.
        :type left1: int
        :param right1: The right boundary of the first image cut.
        :type right1: int
        :param top2: The top boundary of the second image cut.
        :type top2: int
        :param bottom2: The bottom boundary of the second image cut.
        :type bottom2: int
        :param left2: The left boundary of the second image cut.
        :type left2: int
        :param right2: The right boundary of the second image cut.
        :type right2: int
        :return: A tuple containing the adjusted image cuts and their boundaries.
        :rtype: Tuple[np.ndarray, np.ndarray, int, int, int, int, int, int, int, int]
        """
        if top1 == top2:
            if bottom2 > bottom1:
                img1_cut = np.pad(img1_cut[:bottom1, :], ((0, bottom2 - bottom1), (0, 0)), 'constant')
                bottom1 = bottom2
            elif bottom1 > bottom2:
                img2_cut = np.pad(img2_cut[:bottom2, :], ((0, bottom1 - bottom2), (0, 0)), 'constant')
                bottom2 = bottom1
        else:
            if bottom2 - top2 > bottom1 - top1:
                bottom1, top1 = bottom2, top2
            if bottom1 - top1 > bottom2 - top2:
                bottom2, top2 = bottom1, top1

        if left1 == left2:
            if right2 > right1:
                img1_cut = np.pad(img1_cut[:, :right1], ((0, 0), (0, right2 - right1)), 'constant')
                right1 = right2
            elif right1 > right2:
                img2_cut = np.pad(img2_cut[:, :right2], ((0, 0), (0, right1 - right2)), 'constant')
                right2 = right1
        else:
            if right2 - left2 > right1 - left1:
                right1, left1 = right2, left2
            if right1 - left1 > right2 - left2:
                right2, left2 = right1, left1

        return img1_cut, img2_cut, top1, bottom1, left1, right1, top2, bottom2, left2, right2

    def calculate_ratio(self, img1_cut: np.ndarray, img2_cut: np.ndarray) -> float:
        """
        Calculate the ratio of non-zero pixels in the image cuts.

        This method calculates the ratio of non-zero pixels in the given image cuts.

        :param img1_cut: The first image cut as a numpy array.
        :type img1_cut: np.ndarray
        :param img2_cut: The second image cut as a numpy array.
        :type img2_cut: np.ndarray
        :return: The ratio of non-zero pixels in the image cuts.
        :rtype: float
        """
        try:
            ratio = ((len(np.argwhere(img1_cut)) / (img1_cut.shape[0] * img1_cut.shape[1])
                    + len(np.argwhere(img2_cut)) / (img2_cut.shape[0] * img2_cut.shape[1])) / 2)
        except ZeroDivisionError:
            return 0
        return ratio

    def resize_mask(self, i: int, j: int) -> np.ndarray:
        """
        Resize the mask based on the given indices.

        This method resizes the mask by padding it based on the given indices.

        :param i: The vertical index for resizing.
        :type i: int
        :param j: The horizontal index for resizing.
        :return: The resized mask as a numpy array.
        :rtype: np.ndarray
        """
        cutted_area = self.mask[0 - min(0, i):self.mask.shape[0] - max(0, i),
                                0 - min(0, j):self.mask.shape[1] - max(0, j)]
        cutted_area = np.pad(cutted_area, (
            (max(0, i), abs(min(0, i))),
            (max(0, j), abs(min(0, j)))
            ), 'constant')
        return cutted_area

    # def write_ssim_results(self) -> None:
    #     """
    #     Write the resulting images and graphs
    #     """

    #     write_ssim_results(self.m, self.element1, self.element2,
    #                        self.mask, self.ssim_path)



# def write_ssim_results(ssim_m: np.ndarray, element1, element2, mask: np.ndarray,
#                        out_path: str) -> None:
#     """
#     Write the resulting images and graphs for visual inspections

#     :param ssim_m: the ssim matrice containing the results
#     :type ssim_m: np.ndarray
#     :param img1: the first image
#     :type img1: np.ndarray
#     :param name1: the first image name
#     :type name1: str
#     :param img2: the second image
#     :type img2: np.ndarray
#     :param name2: the second image name
#     :type name2: str
#     :param mask: the mask about the region of interest between the images
#     :type mask: np.ndarray
#     :param out_path: the path where to save results
#     :type out_path: str
#     """

#     logging.info("Write SSIM results")

#     indice = np.unravel_index(ssim_m.argmax(), ssim_m.shape)
#     indice = [indice[0]-math.floor(ssim_m.shape[0]/2), indice[1] -
#               math.floor(ssim_m.shape[1]/2)]

#     # draw the plot of the SSIM scores as a heatmap
#     fig, ax = plt.subplots()
#     im = ax.imshow(np.array(ssim_m))
#     plt.savefig(os.path.join(out_path, "ssim_heatmap_" + name1+name2))
#     plt.close()

#     # cut images for drawing
#     img1_cut_original = cut_images(img1, mask)
#     img2_cut_original = cut_images(img2, mask)

#     # import pdb
#     # pdb.set_trace()
#     cutted_area = mask[0-min(0, indice[0]):mask.shape[0] -
#                        max(0, indice[0]), 0-min(0, indice[1]):mask.shape[1] -
#                        max(0, indice[1])]
#     cutted_area = np.pad(cutted_area, ((max(0, indice[0]), abs(min(0, indice[0]))),
#                                        (max(0, indice[1]), abs(min(0, indice[1])))),
#                          'constant')

#     img1_cut = cut_images(img1, mask)
#     img2_cut = cut_images(img2, cutted_area)

#     n = math.floor(ssim_m.shape[0]/2)
#     logging.info(
#         "original_"+str(ssim_m[round(n/2), round(n/2)])+"_"+name1+name2)

#     cv2.imwrite(os.path.join(out_path, "o" +
#                              name1+name2), img1_cut_original+img2_cut_original.
#                 astype(np.uint8))
#     cv2.imwrite(os.path.join(out_path, str(indice)+name1+name2),
#                 img1_cut+img2_cut.astype(np.uint8))


# def cut_images(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     """
#     Cuts an image to the the mask shape

#     :param img: the image to crop
#     :type img: np.ndarray
#     :param mask: the mask with ones corresponding to the area to crop
#     :type mask: np.ndarray
#     :return: the cropped image
#     :rtype: np.ndarray
#     """

#     img_cut = img.copy()
#     img_cut[~mask] = 0

#     try:
#         top, bottom, left, right = get_boundaries(mask)
#     except ValueError:
#         # mask is empty
#         raise

#     # assert that we have the images in the correct boundaries
#     if top < 0:
#         img_cut = np.pad(img_cut, ((-top, 0), (0, 0)), 'constant')
#     if left < 0:
#         img_cut = np.pad(img_cut, ((0, 0), (-left, 0)), 'constant')
#     if bottom > img_cut.shape[0]:
#         img_cut = np.pad(
#             img_cut, ((0, bottom-img_cut.shape[0]), (0, 0)), 'constant')
#     if right > img_cut.shape[1]:
#         img_cut = np.pad(
#             img_cut, ((0, 0), (0, right-img_cut.shape[1])), 'constant')

#     # cut the image
#     img_cut = img_cut[top:bottom+1, left:right+1]

#     return img_cut


