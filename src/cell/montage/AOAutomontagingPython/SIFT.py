import cv2
import numpy as np
from functools import cmp_to_key
try:
    from .helpers import isPixelAnExtremum, \
                    localizeExtremumViaQuadraticFit, \
                    computeKeypointsWithOrientations, \
                    compareKeypoints, \
                    unpackOctave
except ModuleNotFoundError:
    from src.cell.montage.AOAutomontagingPython.helpers import isPixelAnExtremum, \
                    localizeExtremumViaQuadraticFit, \
                    computeKeypointsWithOrientations, \
                    compareKeypoints, \
                    unpackOctave

# code adapted from https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5

float_tolerance = 1e-7

class SIFT:
    """
    This is the SIFT (Scale-Invariant Feature Transform) class. It is used for extracting distinctive
    keypoints from an image that are invariant to image scale, orientation, and affine distortion.
    """
    def __init__(self, image: np.ndarray, sigma: float = 1.6, num_intervals: int = 3,
                 assumed_blur: float = 0.5, image_border_width: int = 5):
        """
        Initialize the SIFT object with an image.

        :param image: The input image
        :type image: np.ndarray
        """
        self.image = image.astype('float32')
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.assumed_blur = assumed_blur
        self.image_border_width = image_border_width
        self.gaussian_kernels = self.generateGaussianKernels()

    # def computeKeypointsAndDescriptors(self) -> tuple:
    #     """
    #     Compute keypoints and their descriptors for the input image.

    #     This method involves several steps:
    #     1. Generate the base image from the input image
    #     2. Compute the number of octaves based on the size of the image
    #     3. Generate Gaussian blurred images for each octave
    #     4. Generate Difference of Gaussian (DoG) images
    #     5. Find scale-space extrema in the DoG images
    #     6. Remove duplicate keypoints
    #     7. Convert the size and location of each keypoint to match the size of the input image
    #     8. Generate the descriptors for each keypoint

    #     :return: The keypoints and their descriptors
    #     :rtype: tuple
    #     """
    #     # import cProfile
    #     # import re
    #     # result = cProfile.runctx('self.computeKeypointsAndDescriptors_()', globals(), locals())

    #     cpu_percents = self.monitor(target=self.computeKeypointsAndDescriptors_)
    #     print(cpu_percents)
    #     print("oe")

    # def monitor(self, target):
    #     import multiprocessing as mp
    #     import psutil
    #     import random
    #     import cv2 as cv
    #     import random
    #     import time
    #     worker_process = mp.Process(target=target)
    #     worker_process.start()
    #     p = psutil.Process(worker_process.pid)

    #     # log cpu usage of `worker_process` every 10 ms
    #     cpu_percents = []
    #     while worker_process.is_alive():
    #         cpu_percents.append(p.cpu_percent())
    #         time.sleep(0.01)

    #     worker_process.join()
    #     return cpu_percents

    def computeKeypointsAndDescriptors(self):
        self.image = self.generateBaseImage()
        num_octaves = self.computeNumberOfOctaves()
        self.gaussian_images = self.generateGaussianImages(num_octaves)
        dog_images = self.generateDoGImages()
        self.findScaleSpaceExtrema(dog_images)
        self.removeDuplicateKeypoints()
        self.convertKeypointsToInputImageSize()
        descriptors = self.generateDescriptors()
        return self.keypoints, descriptors

    def generateGaussianKernels(self) -> np.ndarray:
        """
        Generate Gaussian kernels used for scale-space construction.

        :return: The Gaussian kernels for scale-space construction
        :rtype: np.ndarray
        """
        num_images_per_octave = self.num_intervals + 3
        k = 2 ** (1. / self.num_intervals)
        gaussian_kernels = np.zeros(num_images_per_octave)
        gaussian_kernels[0] = self.sigma

        image_indices = np.arange(1, num_images_per_octave)
        sigma_previous = (k ** (image_indices - 1)) * self.sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[1:] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels

    def generateBaseImage(self) -> np.ndarray:
        """
        Generate the base image for scale-space construction.

        :return: The base image for scale-space construction
        :rtype: np.ndarray
        """
        self.image = cv2.resize(self.image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        sigma_diff = np.sqrt(max((self.sigma ** 2) - ((2 * self.assumed_blur) ** 2), 0.01))
        return cv2.GaussianBlur(self.image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

    def computeNumberOfOctaves(self) -> int:
        """
        Compute the number of octaves for scale-space construction.
        The number of octaves is determined by the size of the image.
        The smaller the image, the fewer octaves can be constructed.

        :return: The number of octaves for scale-space construction
        :rtype: int
        """
        return int(round(np.log2(min(self.image.shape)) - 1))

    def generateGaussianImages(self, num_octaves: int) -> list:
        """
        Generate Gaussian blurred images for each octave in the scale-space.

        :param num_octaves: The number of octaves for scale-space construction
        :type num_octaves: int
        :return: The Gaussian blurred images for each octave
        :rtype: list
        """
        gaussian_images = []
        image = self.image.copy()
        for _ in range(num_octaves):
            gaussian_images_in_octave = [image]
            for gaussian_kernel in self.gaussian_kernels[1:]:
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel,
                                         sigmaY=gaussian_kernel)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-3]
            image = cv2.resize(octave_base,
                               (octave_base.shape[1] // 2, octave_base.shape[0] // 2),
                               interpolation=cv2.INTER_NEAREST)
        return gaussian_images

    def generateDoGImages(self) -> list:
        """
        Generate Difference of Gaussian (DoG) images for each octave in the scale-space.

        :return: The DoG images for each octave
        :rtype: list
        """
        dog_images = []

        for gaussian_images_in_octave in self.gaussian_images:
            dog_images_in_octave = [second_image - first_image for first_image,
                                    second_image in zip(gaussian_images_in_octave,
                                                        gaussian_images_in_octave[1:])]
            dog_images.append(dog_images_in_octave)
        return dog_images

    def findScaleSpaceExtrema(self, dog_images: list, contrast_threshold: float = 0.005):
        """
        Find scale-space extrema in the Difference of Gaussian (DoG) images.

        :param dog_images: The DoG images for each octave
        :type dog_images: list
        :param contrast_threshold: The contrast threshold for detecting extrema
        :type contrast_threshold: float
        """
        threshold = np.floor(0.5 * contrast_threshold / self.num_intervals * 255)
        self.keypoints = []

        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave,
                                                                                       dog_images_in_octave[1:],
                                                                                       dog_images_in_octave[2:])):
                for i in range(self.image_border_width, first_image.shape[0] - self.image_border_width):
                    for j in range(self.image_border_width, first_image.shape[1] - self.image_border_width):
                        if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2],
                                             second_image[i-1:i+2, j-1:j+2],
                                             third_image[i-1:i+2, j-1:j+2],
                                             threshold):
                            localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1,
                                                                                  octave_index, self.num_intervals,
                                                                                  dog_images_in_octave, self.sigma,
                                                                                  contrast_threshold, self.image_border_width)
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = computeKeypointsWithOrientations(keypoint,
                                                                                               octave_index,
                                                                                               self.gaussian_images[octave_index][localized_image_index])
                                self.keypoints.extend(keypoints_with_orientations)

    def removeDuplicateKeypoints(self):
        """
        Sort keypoints and remove duplicate keypoints.
        After this method, the keypoints attribute will only contain unique keypoints.
        """
        if len(self.keypoints) < 2:
            return

        self.keypoints.sort(key=cmp_to_key(compareKeypoints))
        unique_keypoints = [self.keypoints[0]]

        seen = set()
        seen.add(tuple(self.keypoints[0].pt + (self.keypoints[0].size, self.keypoints[0].angle)))

        for next_keypoint in self.keypoints[1:]:
            keypoint_tuple = tuple(next_keypoint.pt + (next_keypoint.size, next_keypoint.angle))
            if keypoint_tuple not in seen:
                seen.add(keypoint_tuple)
                unique_keypoints.append(next_keypoint)

        self.keypoints = unique_keypoints

    def convertKeypointsToInputImageSize(self):
        """
        Convert keypoint point, size, and octave to input image size
        """
        converted_keypoints = []
        for keypoint in self.keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        self.keypoints = converted_keypoints

    def generateDescriptors(self, window_width: int = 5, num_bins: int = 8,
                            scale_multiplier: int = 3, descriptor_max_value: float = 0.5):
        """
        Generate descriptors for each keypoint in the image.

        :param window_width: The width of the window for which the descriptor is computed.
        :type window_width: int, optional
        :param num_bins: The number of bins in the histogram used to compute the descriptor.
        :type num_bins: int, optional
        :param scale_multiplier: The multiplier for the scale of the keypoint to define the width of the descriptor window.
        :type scale_multiplier: int, optional
        :param descriptor_max_value: The maximum value in the descriptor after normalization.
        :type descriptor_max_value: float, optional
        :return: An array of descriptors for each keypoint in the image.
        :rtype: np.ndarray
        """
        descriptors = []

        for keypoint in self.keypoints:
            octave, layer, scale = unpackOctave(keypoint)
            gaussian_image = self.gaussian_images[octave + 1][layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

            # Descriptor window size (described by half_width) follows OpenCV convention
            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                # Smoothing via trilinear interpolation
                # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
                # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
            # Threshold and normalize descriptor_vector
            threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
            # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')

