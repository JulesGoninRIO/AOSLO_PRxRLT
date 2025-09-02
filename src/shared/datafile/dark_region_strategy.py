from abc import ABC, abstractmethod
import numpy as np
import cv2
from skimage import morphology
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Tuple

class DarkRegionsStrategy(ABC):
    """
    Abstract base class for dark regions detection strategies.
    """
    @abstractmethod
    def find_dark_regions(self, image_data: np.ndarray):
        """
        Find dark regions in the given image data.

        :param image_data: The image data to analyze.
        :type image_data: np.ndarray
        """
        pass

class DarkRegionsThresholdingStrategy(DarkRegionsStrategy):
    """
    Strategy for finding dark regions using thresholding.
    """
    def __init__(self):
        """
        Initialize the DarkRegionsThresholdingStrategy with CLAHE parameters.
        """
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    def find_dark_regions(self, image: np.ndarray, save_path: str = None) -> np.ndarray:
        """
        Find dark regions in the image using thresholding.

        :param image: The image to analyze.
        :type image: np.ndarray
        :param save_path: The path to save the binary map, defaults to None.
        :type save_path: str, optional
        :return: The binary map of dark regions.
        :rtype: np.ndarray
        """
        self.equalized = self.apply_clahe(image)
        self.im_blur = self.apply_filters(self.equalized)
        self.bins, self.values = self.smooth_histogram(self.im_blur)
        self.threshold = self.find_threshold(self.bins, self.values)
        self.binary_local = self.create_binary_map(self.im_blur, self.threshold)
        self.save_binary_map(self.binary_local, save_path)
        return self.binary_local

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image.

        :param image: The image to equalize.
        :type image: np.ndarray
        :return: The equalized image.
        :rtype: np.ndarray
        """
        try:
            return self.clahe.apply(image)
        except cv2.error:
            return self.clahe.apply(image.astype(np.uint8))

    def apply_filters(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral and Gaussian filters to the image.

        :param image: The image to filter.
        :type image: np.ndarray
        :return: The filtered image.
        :rtype: np.ndarray
        """
        im_blur = cv2.bilateralFilter(image, 51, 100, 100)
        im_blur = cv2.GaussianBlur(im_blur, (21, 21), 10.0)
        im_blur = cv2.normalize(im_blur, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return (im_blur * 255).astype(np.uint8)

    def smooth_histogram(self, im_blur: np.ndarray) -> tuple:
        """
        Smooth the histogram of the blurred image.

        :param im_blur: The blurred image.
        :type im_blur: np.ndarray
        :return: The smoothed histogram bins and values.
        :rtype: tuple
        """
        hist = np.histogram(im_blur, bins=len(np.unique(im_blur)), range=(0, 255))
        values, bins = hist[0], 0.5 * (hist[1][1:] + hist[1][:-1])
        try:
            smoothed = lowess(values, bins, is_sorted=False, frac=15 / len(values), it=0)
        except ValueError:
            if len(bins) >= 2:
                smoothed = lowess(values, bins, is_sorted=False, frac=1, it=0)
        return smoothed[:, 0], smoothed[:, 1]

    def find_threshold(self, bins: np.ndarray, values: np.ndarray) -> float:
        """
        Find the threshold value for the binary map.

        :param bins: The histogram bins.
        :type bins: np.ndarray
        :param values: The histogram values.
        :type values: np.ndarray
        :return: The threshold value.
        :rtype: float
        """
        indice = self.customized_peak_detection(bins, values)
        try:
            return (bins[indice] + bins[indice + 1]) / 2
        except IndexError:
            return bins[indice]

    def create_binary_map(self, im_blur: np.ndarray, threshold: float) -> np.ndarray:
        """
        Create a binary map from the blurred image using the threshold.

        :param im_blur: The blurred image.
        :type im_blur: np.ndarray
        :param threshold: The threshold value.
        :type threshold: float
        :return: The binary map.
        :rtype: np.ndarray
        """
        binary_local = im_blur > threshold
        return remove_small_dark_dots(binary_local, size=40)

    def save_binary_map(self, binary_map: np.ndarray, save_path: str):
        """
        Save the binary map to the specified path.

        :param binary_map: The binary map to save.
        :type binary_map: np.ndarray
        :param save_path: The path to save the binary map.
        :type save_path: str
        """
        if save_path:
            cv2.imwrite(save_path, 255 * binary_map.astype(np.uint8))

    def find_central_peak(self, bins: np.array, values: np.array, peaks: np.array) -> int:
        """
        Find the central peak in the histogram.

        :param bins: The histogram bins.
        :type bins: np.array
        :param values: The histogram values.
        :type values: np.array
        :param peaks: The detected peaks in the histogram.
        :type peaks: np.array
        :return: The index of the central peak.
        :rtype: int
        """
        central_peaks = [indice for indice in peaks if bins[indice] > np.mean(bins) - np.std(bins)]
        try:
            return [indice for indice in central_peaks if values[indice] == np.max(values[central_peaks])][0]
        except IndexError:
            return peaks[0] if peaks.size > 0 else 0

    def calculate_slopes(self, bins: np.array, values: np.array, central_peak: int):
        """
        Calculate the slopes and second differences of the histogram values.

        :param bins: The histogram bins.
        :type bins: np.array
        :param values: The histogram values.
        :type values: np.array
        :param central_peak: The index of the central peak.
        :type central_peak: int
        :return: The first and second differences of the slopes.
        :rtype: tuple
        """
        slopes = np.diff(values) / np.diff(bins)
        try:
            diff_slopes = np.array([j - i for i, j in zip(slopes[:central_peak - 1], slopes[1:central_peak])])
            second_diff = [(i + j) / 2 for i, j in zip(diff_slopes[:-1], diff_slopes[1:])]
            return diff_slopes, second_diff
        except IndexError:
            return np.array([]), np.array([])

    def find_optimal_indice(self, diff_slopes: np.array, second_diff: np.array, central_peak: int):
        """
        Find the optimal index for thresholding based on the slopes.

        :param diff_slopes: The first differences of the slopes.
        :type diff_slopes: np.array
        :param second_diff: The second differences of the slopes.
        :type second_diff: np.array
        :param central_peak: The index of the central peak.
        :type central_peak: int
        :return: The optimal index for thresholding.
        :rtype: int
        """
        min_found = False
        while not min_found and diff_slopes.size > 0:
            indice = np.argmax(diff_slopes)
            if indice + 1 < len(second_diff) and 0.25 * central_peak <= indice <= 0.75 * central_peak:
                min_found = True
            else:
                diff_slopes[indice] = -np.inf
        return indice if min_found else round(len(diff_slopes) / 2)

    def customized_peak_detection(self, bins: np.array, values: np.array) -> int:
        """
        Perform customized peak detection on the histogram.

        :param bins: The histogram bins.
        :type bins: np.array
        :param values: The histogram values.
        :type values: np.array
        :return: The index of the detected peak.
        :rtype: int
        """
        peaks, _ = find_peaks(values)
        central_peak = self.find_central_peak(bins, values, peaks)
        diff_slopes, second_diff = self.calculate_slopes(bins, values, central_peak)
        if diff_slopes.size > 0:
            indice = self.find_optimal_indice(diff_slopes, second_diff, central_peak)
        else:
            indice = round(len(bins) / 2)
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
