from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict
import cv2
import numpy as np
from scipy.ndimage import shift
from scipy.optimize import differential_evolution, leastsq
from sklearn.metrics import normalized_mutual_info_score

from src.cell.registration.poc import POCFunction, POCModel
from src.shared.helpers.exceptions import ZeroVector, EmptyVector

class RegistrationStrategy(ABC):
    """
    Abstract base class for registration strategies.

    This class defines the interface for image registration strategies.

    :method register: Register two images and return the registration parameters.
    :param ref_image: The reference image as a numpy array.
    :type ref_image: np.ndarray
    :param cmp_image: The comparison image as a numpy array.
    :type cmp_image: np.ndarray
    :return: A tuple containing the registration parameters (dy, dx, match_height).
    :rtype: Tuple[float, float, float]
    """
    @abstractmethod
    def register(self, ref_image: np.ndarray, cmp_image: np.ndarray) -> Tuple[float, float, float]:
        pass

class POCRegistrationStrategy(RegistrationStrategy):
    """
    Phase-Only Correlation (POC) registration strategy.

    This class implements the POC method for registering two images.

    :param FITTING_SHAPE: The shape of the fitting area.
    :type FITTING_SHAPE: Tuple[int, int]
    :param EPS: A small epsilon value for numerical stability.
    :type EPS: float
    """
    FITTING_SHAPE = (8, 8)
    EPS = 0.001

    def register(self, ref_image: np.ndarray, cmp_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Register two images using the POC method.

        :param ref_image: The reference image as a numpy array.
        :type ref_image: np.ndarray
        :param cmp_image: The comparison image as a numpy array.
        :type cmp_image: np.ndarray
        :return: A tuple containing the registration parameters (dy, dx, match_height).
        :rtype: Tuple[float, float, float]
        """
        return self._main_poc_reg(ref_image, cmp_image)

    @staticmethod
    def _main_poc_reg(ref_image: np.ndarray, cmp_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Main POC registration function.

        This method performs the main POC registration process.

        From: 
        High-Accuracy Subpixel Image Registration Based on Phase-Only Correlation
        http://www.aoki.ecei.tohoku.ac.jp/research/docs/e86-a_8_1925.pdf
        https://github.com/alex000kim/ImReg

        :param ref_image: The reference image as a numpy array.
        :type ref_image: np.ndarray
        :param cmp_image: The comparison image as a numpy array.
        :type cmp_image: np.ndarray
        :return: A tuple containing the registration parameters (dy, dx, match_height).
        :rtype: Tuple[float, float, float]
        """
        poc_value = POCFunction(ref_image, cmp_image).evaluate()
        peak = POCRegistrationStrategy._get_peak_position(poc_value, ref_image)
        fitting_area = POCRegistrationStrategy._get_fitting_area(poc_value, peak)
        if fitting_area is None:
            return 0.0, 0.0, 0.0
        y, x = POCRegistrationStrategy._get_grid_positions(peak, ref_image)
        dy, dx, match_height = POCRegistrationStrategy._get_estimates(fitting_area, poc_value, y, x, peak)
        return dy, dx, match_height

    @staticmethod
    def _get_peak_position(poc_value: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
        """
        Get the peak position in the POC value array.

        :param poc_value: The POC value array.
        :type poc_value: np.ndarray
        :param ref_image: The reference image as a numpy array.
        :type ref_image: np.ndarray
        :return: The peak position as a numpy array.
        :rtype: np.ndarray
        """
        max_pos = np.argmax(poc_value)
        return np.array([max_pos / ref_image.shape[1], max_pos % ref_image.shape[1]]).astype(int)

    @staticmethod
    def _get_fitting_area(poc: np.ndarray, peak: np.ndarray) -> np.ndarray:
        """
        Get the fitting area around the peak position.

        :param poc: The POC value array.
        :type poc: np.ndarray
        :param peak: The peak position as a numpy array.
        :type peak: np.ndarray
        :return: The fitting area as a numpy array, or None if the fitting area is invalid.
        :rtype: np.ndarray
        """
        mc = np.array([POCRegistrationStrategy.FITTING_SHAPE[0] / 2.0, POCRegistrationStrategy.FITTING_SHAPE[1] / 2.0])
        fitting_area = poc[
            int(peak[0] - mc[0]): int(peak[0] + mc[0] + 1),
            int(peak[1] - mc[1]): int(peak[1] + mc[1] + 1)]
        if fitting_area.shape != (POCRegistrationStrategy.FITTING_SHAPE[0] + 1, POCRegistrationStrategy.FITTING_SHAPE[1] + 1):
            return None
        return fitting_area

    @staticmethod
    def _get_grid_positions(peak: np.ndarray, ref_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the grid positions for the fitting area.

        :param peak: The peak position as a numpy array.
        :type peak: np.ndarray
        :param ref_image: The reference image as a numpy array.
        :type ref_image: np.ndarray
        :return: A tuple containing the y and x grid positions as numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        mc = np.array([POCRegistrationStrategy.FITTING_SHAPE[0] / 2.0, POCRegistrationStrategy.FITTING_SHAPE[1] / 2.0])
        m = np.array([ref_image.shape[0] / 2.0, ref_image.shape[1] / 2.0])
        y, x = np.mgrid[-mc[0]:mc[0] + 1, -mc[1]:mc[1] + 1]
        y = np.ceil(y + peak[0] - m[0])
        x = np.ceil(x + peak[1] - m[1])
        return y, x

    @staticmethod
    def _get_estimates(fitting_area: np.ndarray, poc: np.ndarray, y: np.ndarray, x: np.ndarray, peak: np.ndarray) -> Tuple[float, float, float]:
        """
        Get the registration estimates.

        :param fitting_area: The fitting area as a numpy array.
        :type fitting_area: np.ndarray
        :param poc: The POC value array.
        :type poc: np.ndarray
        :param y: The y grid positions as a numpy array.
        :type y: np.ndarray
        :param x: The x grid positions as a numpy array.
        :type x: np.ndarray
        :param peak: The peak position as a numpy array.
        :type peak: np.ndarray
        :return: A tuple containing the registration estimates (dy, dx, match_height).
        :rtype: Tuple[float, float, float]
        """
        m = np.array([fitting_area.shape[0] / 2.0, fitting_area.shape[1] / 2.0])
        u = (m / 2)
        p0 = np.array([0.0, -(peak[0] - m[0]) - POCRegistrationStrategy.EPS, -(peak[1] - m[1]) - POCRegistrationStrategy.EPS])
        error_func = lambda p: np.ravel(POCModel(p[0], p[1], p[2], poc, u).evaluate(y, x) - fitting_area)
        estimate = leastsq(error_func, p0)
        match_height = estimate[0][0]
        dx = -estimate[0][1]
        dy = -estimate[0][2]
        return dy, dx, match_height

class MIRegistrationStrategy(RegistrationStrategy):
    """
    Mutual Information (MI) registration strategy.

    This class implements the MI method for registering two images.

    :param bounds: The bounds for the optimization algorithm.
    :type bounds: List[Tuple[int, int]]
    """
    bounds = [(-10, 10), (-10, 10)]

    def register(self, ref_image: np.ndarray, cmp_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Register two images using the MI method.

        :param ref_image: The reference image as a numpy array.
        :type ref_image: np.ndarray
        :param cmp_image: The comparison image as a numpy array.
        :type cmp_image: np.ndarray
        :return: A tuple containing the registration parameters (dx, dy, match_height).
        :rtype: Tuple[float, float, float]
        """
        return MIRegistrationStrategy._main_mi_reg(ref_image, cmp_image)

    @staticmethod
    def _main_mi_reg(ref_image: np.ndarray, cmp_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Correlator based on Mutual Information Algorithm
        http://www.sci.utah.edu/~fletcher/CS7960/slides/Yen-Yun.pdf

        :param ref_image: The reference image data.
        :type ref_image: np.ndarray
        :param cmp_image: The comparison image data.
        :type cmp_image: np.ndarray
        :return: A tuple with the residual in X, residual in Y, and the match height.
        :rtype: Tuple[float, float, float]
        """
        def obj_func(dx_dy):
            return -MIRegistrationStrategy.nmi(shift(ref_image, dx_dy), cmp_image)

        opt_res = differential_evolution(obj_func, MIRegistrationStrategy.bounds)
        (dx, dy), match_height = -opt_res.x, -opt_res.fun

        return dx, -dy, match_height

    @staticmethod
    def nmi(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the Normalized Mutual Information (NMI) score of two arrays.

        :param a: The first array.
        :type a: np.ndarray
        :param b: The second array.
        :type b: np.ndarray
        :raise AssertionError: If the shapes of the arrays are not the same.
        :raise EmptyVector: If one of the arrays is empty.
        :return: The NMI score.
        :rtype: float
        """
        assert a.shape == b.shape, "The two arrays must have the same shape"
        if len(a) == 0:
            raise EmptyVector(f"{a} is empty")
        if len(b) == 0:
            raise EmptyVector(f"{b} is empty")
        return normalized_mutual_info_score(a.flatten(), b.flatten())

class ECCRegistrationStrategy(RegistrationStrategy):
    """
    Enhanced Correlation Coefficient (ECC) registration strategy.

    This class implements the ECC method for registering two images.
    """

    def register(self, ref_image: np.ndarray, cmp_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Register two images using the ECC method.

        :param ref_image: The reference image as a numpy array.
        :type ref_image: np.ndarray
        :param cmp_image: The comparison image as a numpy array.
        :type cmp_image: np.ndarray
        :return: A tuple containing the registration parameters (dx, dy, ecc_value).
        :rtype: Tuple[float, float, float]
        """
        return self._main_ecc_reg(ref_image, cmp_image)

    @staticmethod
    def _initialize_warp_matrix(warp_mode: int) -> np.ndarray:
        """
        Initialize the warp matrix based on the motion model.

        :param warp_mode: The motion model.
        :type warp_mode: int
        :return: Initialized warp matrix.
        :rtype: np.ndarray
        """
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            return np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            warp_matrix[0, 2] = 12
            warp_matrix[1, 2] = -293
            return warp_matrix

    @staticmethod
    def _main_ecc_reg(ref_image: np.ndarray, cmp_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute ECC registration.

        The ECC algorithm is a direct (gradient-based) image registration algorithm.
        Due to gradient information, it achieves high accuracy in parameter estimation
        (i.e., subpixel accuracy). Its performance is invariant to global illumination
        changes in images since it considers the correlation coefficient (zero-mean
        normalized cross-correlation) as an objective function.

        :param ref_image: The reference image.
        :type ref_image: np.ndarray
        :param cmp_image: The comparison image.
        :type cmp_image: np.ndarray
        :return: A tuple containing the offset in the x direction, the offset in the y direction, and the value of the ECC function.
        :rtype: Tuple[float, float, float]
        """
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = ECCRegistrationStrategy._initialize_warp_matrix(warp_mode)
        number_of_iterations = 100000
        termination_eps = 1e-10
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        try:
            (cc, warp_matrix) = cv2.findTransformECC(ref_image, cmp_image, warp_matrix, warp_mode, criteria)
        except cv2.error:
            return 0, 0, 0

        return warp_matrix[0, 2], -warp_matrix[1, 2], cc
