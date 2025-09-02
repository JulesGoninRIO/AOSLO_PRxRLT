from http.client import PRECONDITION_FAILED
from typing import Dict, List, Tuple, Callable, Iterable
from collections import defaultdict
import warnings
import sys
from enum import Enum, auto
import math
from pathlib import Path
from shapely import Point
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, root_scalar
import scipy.optimize
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import cv2

from deprecated import deprecated

from src.cell.analysis.density import Density
# from src.cell.cell_detection.cone import Cone
from src.shared.helpers.direction import Direction
from src.cell.analysis.constants import AREA_THRESHOLD, MM_PER_DEGREE, PIXELS_PER_DEGREE
from src.cell.analysis.helpers import eccentricity, sort_by_ecc
from src.cell.cell_detection.constants import IMAGE_SIZE
# from src.cell.cell_detection.cone_gatherer import ConeGatherer
from src.cell.montage.montage_mosaic import MontageMosaic
from src.cell.processing_path_manager import ProcessingPathManager
from src.configs.parser import Parser
from src.shared.computer_vision.point import Point
from src.plotter.cone_mosaic_plotter import ConeMosaicPlotter
from src.plotter.density_plotter import DensityPlotter

class PerDistanceCalculator:
    """
    A class to calculate distances and densities per distance unit.

    :ivar step: Step size for rounding.
    :vartype step: float
    :ivar round_step: Rounded step size.
    :vartype round_step: int
    :ivar factor: Density growing factor.
    :vartype factor: float
    """
    def __init__(self):
        """
        Initialize the PerDistanceCalculator with step size and density growing factor.
        """
        self.step = 0.1
        self.round_step = round(self.step * 10)
        try:
            self.factor = Parser.get_density_growing_factor()
        except AttributeError:
            Parser.initialize()
            self.factor = Parser.get_density_growing_factor()

    def round_distance(self, distances: np.ndarray) -> float:
        """
        Round the distances to the nearest step size.

        :param distances: Array of distances in pixels.
        :type distances: np.ndarray
        :return: Rounded distance in degrees.
        :rtype: float
        """
        return np.round(distances / PIXELS_PER_DEGREE, self.round_step)

    def unround_distance(self, rounded_distance: float) -> Tuple[float, float]:
        """
        Convert the rounded distance back to pixel limits.

        :param rounded_distance: Rounded distance in degrees.
        :type rounded_distance: float
        :return: Tuple containing the lower and upper pixel limits.
        :rtype: Tuple[float, float]
        """
        degree_lower_limit = rounded_distance - (0.5 * 10**(-self.round_step))
        degree_upper_limit = rounded_distance + (0.5 * 10**(-self.round_step))
        pixel_lower_limit = degree_lower_limit * PIXELS_PER_DEGREE
        pixel_upper_limit = degree_upper_limit * PIXELS_PER_DEGREE
        return pixel_lower_limit, pixel_upper_limit

class ConeNumberPerDistanceCalculator(PerDistanceCalculator):
    """
    A class to calculate the number of cones per distance unit from a given center point.

    :param center: The center point from which distances are calculated.
    :type center: Point
    """
    def __init__(self, center: Point):
        """
        Initialize the ConeNumberPerDistanceCalculator with a center point.

        :param center: The center point from which distances are calculated.
        :type center: Point
        """
        super().__init__()
        self.pixels_per_degree = PIXELS_PER_DEGREE
        self.center = center
        self.growing_factor = lambda x: self.factor[0] * x + self.factor[1] / PIXELS_PER_DEGREE

    def get_counts(self, cones: List[List[int]], binary_map: np.ndarray = None) -> Tuple[Dict[float, int]]:
        """
        Get the counts and locations of cones per distance unit in both X and Y directions.

        :param cones: List of cone coordinates.
        :type cones: List[List[int]]
        :param binary_map: Optional binary map to filter cones.
        :type binary_map: np.ndarray, optional
        :return: Tuple containing dictionaries of cone counts and locations per distance unit in X and Y directions.
        :rtype: Tuple[Dict[float, int], Dict[float, int], Dict[float, List[Tuple[int, int]]], Dict[float, List[Tuple[int, int]]]]
        """
        if len(cones) == 0:
            return defaultdict(int), defaultdict(int)

        number_cones_per_distance_x: Dict[float, int] = defaultdict(int)
        number_cones_per_distance_y: Dict[float, int] = defaultdict(int)
        cones_location_per_distance_x: Dict[float, List[Tuple[int, int]]] = defaultdict(lambda: [])
        cones_location_per_distance_y: Dict[float, List[Tuple[int, int]]] = defaultdict(lambda: [])

        cones = np.array(cones)
        distance_x = cones[:, 0] - self.center.x
        distance_y = cones[:, 1] - self.center.y
        distances = np.sqrt(distance_x**2 + distance_y**2)

        distance_rounded_x = self.round_distance(distance_x)
        distance_rounded_y = self.round_distance(distance_y)
        distance_to_center_rounded = self.round_distance(distances)

        abs_distance_x = np.abs(distance_rounded_x)
        abs_distance_y = np.abs(distance_rounded_y)

        for i in range(len(cones)):
            x, y = cones[i]
            if binary_map is not None and not binary_map[y, x]:
                continue
            if abs_distance_y[i] <= self.growing_factor(abs_distance_x[i]):
                if distance_x[i] < 0:
                    number_cones_per_distance_x[-distance_to_center_rounded[i]] += 1
                    cones_location_per_distance_x[-distance_to_center_rounded[i]].append((x, y))
                else:
                    number_cones_per_distance_x[distance_to_center_rounded[i]] += 1
                    cones_location_per_distance_x[distance_to_center_rounded[i]].append((x, y))
            if abs_distance_x[i] <= self.growing_factor(abs_distance_y[i]):
                if distance_y[i] < 0:
                    number_cones_per_distance_y[-distance_to_center_rounded[i]] += 1
                    cones_location_per_distance_y[-distance_to_center_rounded[i]].append((x, y))
                else:
                    number_cones_per_distance_y[distance_to_center_rounded[i]] += 1
                    cones_location_per_distance_y[distance_to_center_rounded[i]].append((x, y))

        return number_cones_per_distance_x, number_cones_per_distance_y, cones_location_per_distance_x, cones_location_per_distance_y

class OriginalPixelPerDistanceCalculator(PerDistanceCalculator):
    """
    A class to calculate the number of pixels per distance unit from a given center point.

    :param min_location: Minimum location value.
    :type min_location: float
    :param max_location: Maximum location value.
    :type max_location: float
    :param center: The center point from which distances are calculated.
    :type center: Point
    """
    def __init__(self, min_location, max_location, center):
        """
        Initialize the OriginalPixelPerDistanceCalculator with min and max locations and a center point.

        :param min_location: Minimum location value.
        :type min_location: float
        :param max_location: Maximum location value.
        :type max_location: float
        :param center: The center point from which distances are calculated.
        :type center: Point
        """
        super().__init__()
        adjusted_min, adjusted_max = min_location - self.step / 2 if min_location < 0 else min_location, max_location + self.step / 2
        self.locations = PIXELS_PER_DEGREE * np.arange(adjusted_min, adjusted_max, step=self.step)
        self.center = center
        self.area_per_locations_x = {}
        self.area_per_locations_y = {}
        self.area_per_locations = {}
        self.growing_factor = lambda x: self.factor[0] * x + self.factor[1] / PIXELS_PER_DEGREE

    def get_counts(self, binary_map: np.ndarray, direction: str) -> Dict[float, int]:
        """
        Get the counts of pixels per distance unit in the specified direction.

        :param binary_map: Binary map to filter pixels.
        :type binary_map: np.ndarray
        :param direction: Direction to calculate counts ('x' or 'y').
        :type direction: str
        :return: Dictionary of pixel counts per distance unit.
        :rtype: Dict[float, int]
        :raises ValueError: If the direction is not 'x' or 'y'.
        """
        if direction.lower() not in ['x', 'y']:
            raise ValueError("Direction must be either 'x' or 'y'")

        area_per_locations = self.calculate_area_per_location(binary_map, direction)
        return area_per_locations

    def calculate_area_per_location(self, binary_map: np.ndarray, direction: str) -> Dict[float, int]:
        """
        Calculate the area per location in the specified direction.

        :param binary_map: Binary map to filter pixels.
        :type binary_map: np.ndarray
        :param direction: Direction to calculate area ('x' or 'y').
        :type direction: str
        :return: Dictionary of areas per location.
        :rtype: Dict[float, int]
        """
        center = [self.center.x, self.center.y] if direction == 'x' else [self.center.y, self.center.x]
        shape = binary_map.shape[1] if direction == 'x' else binary_map.shape[0]
        direction_index = 0 if direction == 'x' else 1

        # Pre-compute distances from center and their rounded versions
        distances_from_center = np.abs(np.arange(shape) - center[direction_index])
        growing_factor_distances = self.growing_factor(distances_from_center)
        max_distance = np.max(distances_from_center) + self.step * PIXELS_PER_DEGREE
        distances_from_center_rounded = self.round_distance(distances_from_center)
        all_values_set = set(distances_from_center_rounded).union(-distances_from_center_rounded)
        area_per_locations = {distance: 0 for distance in all_values_set}

        # Vectorized computation for distance checks
        col_indices = np.arange(shape)
        for row_index in range(binary_map.shape[direction_index]):
            col_range = col_indices[(col_indices >= max(0, center[1] - round(growing_factor_distances[row_index]))) &
                                    (col_indices <= min(center[1] + round(growing_factor_distances[row_index]), shape - 1))]
            for col_index in col_range:
                if binary_map[row_index, col_index] == 0:
                    continue
                distance = np.sqrt(distances_from_center[col_index]**2 + distances_from_center[row_index]**2)
                if distance > max_distance:
                    continue
                closest_distance = OriginalPixelPerDistanceCalculator.find_closest(distances_from_center_rounded, self.round_distance(distance))
                if direction == 'x' and row_index < center[0] or direction == 'y' and col_index < center[0]:
                    closest_distance = -closest_distance
                area_per_locations[closest_distance] += 1

        return area_per_locations

    @staticmethod
    def count_pixels(area, binary_map):
        """
        Count the number of non-zero pixels in the specified area of the binary map.

        :param area: Area to count pixels in.
        :type area: list
        :param binary_map: Binary map to filter pixels.
        :type binary_map: np.ndarray
        :return: Number of non-zero pixels in the area.
        :rtype: int
        """
        if len(area) == 0:
            return 0
        indices = np.array(area)
        return np.count_nonzero(binary_map[indices[:, 0], indices[:, 1]])

    @staticmethod
    def find_closest(distances, target):
        """
        Find the value in distances that is closest to the target.

        :param distances: List of distances.
        :type distances: list
        :param target: Target distance.
        :type target: float
        :return: Closest distance value.
        :rtype: float
        """
        closest_value = min(distances, key=lambda x: abs(x - target))
        return closest_value

class RefactoredPixelPerDistanceCalculator(PerDistanceCalculator):
    """
    A class to calculate the number of pixels per distance unit from a given center point.

    :param min_location: Minimum location value.
    :type min_location: float
    :param max_location: Maximum location value.
    :type max_location: float
    :param center: The center point from which distances are calculated.
    :type center: Point
    """
    def __init__(self, min_location, max_location, center):
        """
        Initialize the RefactoredPixelPerDistanceCalculator with min and max locations and a center point.

        :param min_location: Minimum location value.
        :type min_location: float
        :param max_location: Maximum location value.
        :type max_location: float
        :param center: The center point from which distances are calculated.
        :type center: Point
        """
        super().__init__()
        adjusted_min, adjusted_max = min_location - self.step / 2 if min_location < 0 else min_location, max_location + self.step / 2
        self.locations = PIXELS_PER_DEGREE * np.arange(adjusted_min, adjusted_max, step=self.step)
        self.center = center
        self.area_per_locations_x = {}
        self.area_per_locations_y = {}
        self.area_per_locations = {}
        self.growing_factor = lambda x: self.factor[0] * x + self.factor[1]

    def get_counts(self, binary_map: np.ndarray, direction: str) -> Dict[float, int]:
        """
        Get the counts of pixels per distance unit in the specified direction.

        :param binary_map: Binary map to filter pixels.
        :type binary_map: np.ndarray
        :param direction: Direction to calculate counts ('x' or 'y').
        :type direction: str
        :return: Dictionary of pixel counts per distance unit.
        :rtype: Dict[float, int]
        :raises ValueError: If the direction is not 'x' or 'y'.
        """
        if direction.lower() not in ['x', 'y']:
            raise ValueError("Direction must be either 'x' or 'y'")

        area_per_locations = self.calculate_area_per_location(binary_map, direction)
        return area_per_locations

    def calculate_area_per_location(self, binary_map: np.ndarray, direction: str) -> Dict[float, int]:
        """
        Calculate the area per location in the specified direction.

        :param binary_map: Binary map to filter pixels.
        :type binary_map: np.ndarray
        :param direction: Direction to calculate area ('x' or 'y').
        :type direction: str
        :return: Dictionary of areas per location.
        :rtype: Dict[float, int]
        """
        import time
        start = time.time()
        center = [self.center.x, self.center.y]
        shape = binary_map.shape[1] if direction == 'x' else binary_map.shape[0]
        width = binary_map.shape[0] if direction == 'x' else binary_map.shape[1]
        direction_index = 0 if direction == 'x' else 1
        other_direction_index = 1 if direction == 'x' else 0

        # Vectorized distance calculation
        distances_from_center = np.abs(np.arange(shape) - center[direction_index])
        other_distance_from_center = np.abs(np.arange(width) - center[other_direction_index])
        growing_factor_distances = self.growing_factor(distances_from_center)
        other_growing_factor_distances = self.growing_factor(other_distance_from_center)
        distances_from_center_rounded = self.round_distance(distances_from_center)
        all_values_set = set(distances_from_center_rounded).union(-distances_from_center_rounded)
        area_per_locations = {distance: 0 for distance in all_values_set}

        # Pre-compute a mask for valid pixels based on growing factor and max distance
        valid_pixels_mask = self.compute_valid_pixels_mask(binary_map, center, growing_factor_distances, direction)
        valid_indices = np.where(valid_pixels_mask)  # y, x

        valid_indices = np.array(valid_indices)

        if direction == 'x':
            col_distances = distances_from_center[valid_indices[1]]**2
            row_distances = other_distance_from_center[valid_indices[0]]**2
        else:
            col_distances = distances_from_center[valid_indices[0]]**2
            row_distances = other_distance_from_center[valid_indices[1]]**2
        distances = np.sqrt(col_distances + row_distances)

        # Use the vectorized find_closest method
        closest_distances = RefactoredPixelPerDistanceCalculator.vectorized_find_closest(distances_from_center_rounded, self.round_distance(distances))

        if direction == 'x':
            is_x_direction = True
            below_center_x = valid_indices[1] < center[0]
            adjust_sign = is_x_direction and below_center_x
        else:
            is_x_direction = False
            below_center_y = valid_indices[0] < center[1]
            adjust_sign = not is_x_direction and below_center_y

        adjusted_distances = np.where(adjust_sign == 1, -closest_distances, closest_distances)

        # Efficiently count occurrences of each unique closest_distance
        unique_distances, counts = np.unique(adjusted_distances, return_counts=True)
        area_per_locations = dict(zip(unique_distances, counts))

        return area_per_locations

    @staticmethod
    def compute_valid_pixels_mask(binary_map: np.ndarray, center: list, growing_factor_distances: np.ndarray, direction: str) -> np.ndarray:
        """
        Compute a mask of valid pixels based on the growing factor distances.

        :param binary_map: Binary map to filter pixels.
        :type binary_map: np.ndarray
        :param center: The center point from which distances are calculated.
        :type center: list
        :param growing_factor_distances: Array of growing factor distances.
        :type growing_factor_distances: np.ndarray
        :param direction: Direction to calculate the mask ('x' or 'y').
        :type direction: str
        :return: Mask of valid pixels.
        :rtype: np.ndarray
        """
        valid_pixels_mask = np.zeros(binary_map.shape, dtype=bool)

        # Compute the grid of indices
        rows, cols = np.indices(binary_map.shape)

        # Calculate distances from the center for both axes
        if direction == 'x':
            distances_from_center_x = np.abs(cols - center[0])
            distances_from_center_y = np.abs(rows - center[1])
        else:  # direction == 'y'
            distances_from_center_x = np.abs(cols - center[0])
            distances_from_center_y = np.abs(rows - center[1])

        # Apply growing factor to determine valid range in the primary direction
        if direction == 'x':
            valid_range_mask = distances_from_center_y <= growing_factor_distances[None, :]
        else:
            valid_range_mask = distances_from_center_x <= growing_factor_distances[:, None]

        # Combine conditions: within growing factor range and less than max distance
        valid_pixels_mask = valid_range_mask & binary_map.astype(bool)

        return valid_pixels_mask

    @staticmethod
    def count_pixels(area, binary_map):
        """
        Count the number of non-zero pixels in the specified area of the binary map.

        :param area: Area to count pixels in.
        :type area: list
        :param binary_map: Binary map to filter pixels.
        :type binary_map: np.ndarray
        :return: Number of non-zero pixels in the area.
        :rtype: int
        """
        if len(area) == 0:
            return 0
        indices = np.array(area)
        return np.count_nonzero(binary_map[indices[:, 0], indices[:, 1]])

    @staticmethod
    def find_closest(distances, target):
        """
        Find the value in distances that is closest to the target.

        :param distances: List of distances.
        :type distances: list
        :param target: Target distance.
        :type target: float
        :return: Closest distance value.
        :rtype: float
        """
        closest_value = min(distances, key=lambda x: abs(x - target))
        return closest_value

    @staticmethod
    def vectorized_find_closest(distances_from_center_rounded, distances):
        """
        Find the closest values in distances_from_center_rounded for each distance in distances.

        :param distances_from_center_rounded: Array of rounded distances from center.
        :type distances_from_center_rounded: np.ndarray
        :param distances: Array of distances to find closest values for.
        :type distances: np.ndarray
        :return: Array of closest values.
        :rtype: np.ndarray
        """
        distances_from_center_rounded = np.asarray(distances_from_center_rounded)
        distances = np.asarray(distances)
        closest_values = np.empty_like(distances)
        for i, distance in enumerate(distances):
            abs_diff = np.abs(distances_from_center_rounded - distance)
            min_index = np.argmin(abs_diff)
            closest_values[i] = distances_from_center_rounded[min_index]
        return closest_values

class YellottConeDensityCalculator:
    """
    A class to calculate cone densities using the Yellott's ring-based method.

    :param path_manager: The manager for processing paths.
    :type path_manager: ProcessingPathManager
    :param montage_mosaic: The montage mosaic data.
    :type montage_mosaic: MontageMosaic
    """
    type Model = Callable[[float | np.ndarray], float | np.ndarray]
    type ParamModel = Callable[[float | np.ndarray, Iterable], float | np.ndarray]

    def __init__(self, path_manager: ProcessingPathManager, montage_mosaic: MontageMosaic, step: float = 0.1):
        """
        Initialize the YellottConeDensityCalculator with the given path manager and montage mosaic.

        :param path_manager: The manager for processing paths.
        :type path_manager: ProcessingPathManager
        :param montage_mosaic: The montage mosaic data.
        :type montage_mosaic: MontageMosaic
        """
        self.path_manager = path_manager
        self.montage_mosaic = montage_mosaic
        self.step = step
        self.round_step = round(self.step * 10)
        self.right_eye = True

    def _find_inflection_point(self, x, y) -> int:
        d1y_dx1 = np.gradient(y, x)
        d2y_dx2 = np.gradient(d1y_dx1, x)
        inflexion_points = np.where(np.diff(np.sign(d2y_dx2)))[0]
        # keep only the inflection points of positive third derivative
        inflexion_points = inflexion_points[np.gradient(d2y_dx2, x)[inflexion_points] > 0]
        # keep only the inflection point with most negative first derivative
        return x[inflexion_points[np.argmin(d1y_dx1[inflexion_points])]]
    
    def _fitted_linear_l1(self, x: np.ndarray, y: np.ndarray, p0: np.ndarray) -> np.ndarray:
        lin = lambda x, p: p[0] - p[1] * x
        popt, _ = self._fit_curve_l1(x, y, lin, p0, f_scale=5)
        return lin(x, popt)
        
    def _fit_curve_l1(self,
                   x: np.ndarray,
                   y: np.ndarray,
                   model: ParamModel, 
                   p0: np.ndarray,
                   lower_bound: np.ndarray | List | None = None,
                   upper_bound: np.ndarray | List | None = None,
                   f_scale: float = 3000) -> Tuple[np.ndarray, np.ndarray]:
        if lower_bound is None:
            lower_bound = np.zeros_like(p0)
        if upper_bound is None:
            upper_bound = np.inf
        residuals_fun = lambda p, x, y: model(x, p) - y
        results = least_squares(residuals_fun, p0, args=(x, y), loss='soft_l1', f_scale=f_scale, bounds=(lower_bound, upper_bound))
        return results.x, results.fun

    def get_density_of_image(self, img: np.ndarray, eccentricity: float, resize_factor: float = 1) -> float:
        """
        Get the cone density of the given image.
        See https://doi.org/10.1167/tvst.8.5.26 for more information about the method.
        """
        img_resized = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        # compute the power spectrum of the image
        dft = np.fft.fftshift(np.fft.fft2(img_resized))
        power_spectrum = np.log10(np.abs(dft)**2)
        power_spectrum = 255 * (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())

        # extract the polar image and compute the average profile
        rows, cols = power_spectrum.shape
        center = (cols // 2, rows // 2)
        max_radius = min(center)
        angles_to_try = [
            (np.arange(70,110).reshape(1,-1)  + 180 * np.arange(2).reshape(-1,1)).flatten(), # vertical
            (np.arange(-20,20).reshape(1,-1)  + 180 * np.arange(2).reshape(-1,1)).flatten(), # horizontal
            (np.arange(30,60).reshape(1,-1)  + 90 * np.arange(4).reshape(-1,1)).flatten() # diagonals
        ]
        densities = []
        for angles in angles_to_try:
            polar_image = np.zeros((max_radius, angles.size), dtype=np.float32)
            for r in range(max_radius):
                for i, theta in enumerate(angles):
                    x = int(center[0] + r * np.cos(np.radians(theta)))
                    y = int(center[1] + r * np.sin(np.radians(theta)))
                    if 0 <= x < cols and 0 <= y < rows:
                        polar_image[r, i] = power_spectrum[y, x]
            average_profile = np.mean(polar_image, axis=1)
            
            idx_cut = min(5, np.ceil(len(average_profile) / 20).astype(int))
            average_profile_cut = average_profile[idx_cut:]
            average_profile_smooth = gaussian_filter1d(average_profile_cut, 5.5)
            x = idx_cut + np.arange(len(average_profile_cut))
            lim = int(min(self._find_inflection_point(x, average_profile_smooth), len(average_profile_cut)/2))
            x_data = x[:lim]
            average_profile_cut = average_profile_cut[:lim]
            # interpolate linearly between the first and last point
            interp = np.interp(x_data, [x_data[0], x_data[-1]], [average_profile_smooth[0], average_profile_smooth[lim-1]])
            diff_cut = average_profile_cut - interp
            linear_diff_cut = self._fitted_linear_l1(x_data, diff_cut, [10, 0.5])
            # second_half_start = len(x_data) // 2
            diff_diff_cut = diff_cut - linear_diff_cut
            peaks, _ = find_peaks(diff_diff_cut)
            if peaks.size > 0:
                largest_peak = peaks[np.argmax(diff_diff_cut[peaks])]
            else:
                densities.append(0)
                continue

            yellott_radius_pixels = x_data[largest_peak]
            density = np.sqrt(3) / 2 * (resize_factor * yellott_radius_pixels / img_resized.shape[0] * PIXELS_PER_DEGREE / MM_PER_DEGREE)**2
            densities.append(density)

        density = np.median(densities)
        if density < 5000 + 45000 * np.exp(-1 * np.abs(eccentricity)):
            density = np.max(densities)
        return density
    
    def get_densities(self) -> Tuple[np.ndarray, np.ndarray]:
        densities = defaultdict(lambda: [])
        mosaic_center = self.montage_mosaic.transformed_center

        for element in self.montage_mosaic.montage_elements:
            img_file = element.image_file
            if img_file.eye != 'OD':
                self.right_eye = False
            directions = {Direction.X: -1 <= img_file.y_position <= 1, Direction.Y: -1 <= img_file.x_position <= 1}
            if not any(directions.values()):
                continue
            dark_regions = img_file.dark_regions
            img_array = cv2.convertScaleAbs(img_file.data, alpha=2)

            # adaptive crop size based on eccentricity (see grahically https://www.desmos.com/calculator/m9r0mbb4me)

            transformed_elem_loc = element.transform.transform_points(np.array([IMAGE_SIZE,IMAGE_SIZE])/2).squeeze().astype(int) # location of the image center after transformation
            distance = eccentricity(transformed_elem_loc, None, mosaic_center)
            crop_size = int(min(300, 160 * math.sqrt(1 + distance ** 2 / 20))) # smaller crop size for smaller eccentricities, larger for larger eccentricities
            center_offset = crop_size / 2
            ideal_overlap = 1/3 # ratio of overlap size / crop size (ideal in the sense wanted by the user, but not necessarily perfectly achievable)
            
            num_crops = int(math.ceil(1 + (IMAGE_SIZE / crop_size - 1) / (1 - ideal_overlap)))
            actual_overlap = 1 - (IMAGE_SIZE / crop_size - 1) / (num_crops - 1) # actual overlap is the one that makes the number of crops fit the image size with minimal leftover pixels
            overlap_px = int(crop_size * (1 - actual_overlap))
            _coords = np.arange(0, IMAGE_SIZE - crop_size + 1, overlap_px)
            crop_indices = np.array(np.meshgrid(_coords, _coords)).T.reshape(-1, 2).astype(int)

            transformed_crop_indices = element.transform.transform_points(crop_indices + center_offset)

            for coords, trans_coords in zip(crop_indices, transformed_crop_indices):
                if dark_regions[coords[1]:coords[1]+crop_size, coords[0]:coords[0]+crop_size].sum() < 0.8 * crop_size ** 2:
                    continue
                distance = eccentricity(trans_coords, None, mosaic_center)
                density = self.get_density_of_image(img_array[coords[1]:coords[1]+crop_size, coords[0]:coords[0]+crop_size], distance, resize_factor=1)
                if density == 0 or density > 150_000:
                    continue
                for direction, pos in directions.items():
                    if pos:
                        densities[direction].append([eccentricity(trans_coords, direction, mosaic_center, self.right_eye), density])

        return sort_by_ecc(densities[Direction.X]), sort_by_ecc(densities[Direction.Y])

    def fit_densities(self, densities: np.ndarray, density_plotter: DensityPlotter, direction: Direction, target = None) -> Tuple[np.ndarray, np.ndarray, Model, float, float, float, float, float]:
        def __model(x, p):
            #exponent of the model, the real model should be exp(__model(x, p))
            X = np.abs(x - p[4])
            return np.where(x < p[4],
                            p[0] - p[1] * X + p[3] / (X + p[2]),
                            p[5] - p[6] * X + p[8] / (X + p[7]))

        def __loss(p, x, y_data):
            """
            Assumptions:
            - data points are less reliable for smaller x
            - data points underestimates the true distribution for smaller x
            - data points may overestimate the true distribution for larger x
            """
            
            soft_l1 = lambda x, C=1: 2 * C * (np.sqrt(1 + (x / C) ** 2) - 1)
            weight_periph = 1 - np.exp(-5 * np.abs(x - p[4]))  # gives importance to non-central data
            weight_overest = 1 / (1 + np.exp(-5 * (np.abs(x - p[4]) - 1))) # =0 for central data
            weight_underest = 1 - weight_overest # =0 for peripheral data
            residuals = y_data - __model(x, p)
            C = 0.3
            loss = np.mean(weight_periph * (
                weight_overest * soft_l1(np.where(residuals <  C, residuals, 0), C)
                + weight_underest * soft_l1(np.where(residuals > -C, residuals, 0), C)
            ))
            return loss

        def __fit(c, x_data, y_data):
            if isinstance(c, np.ndarray):
                c = c.item()
            
            

            # constraints on the model:
            #    - value of the peak is constrained to be between 145_000 and 320_000:
            peak_val = lambda p: p[0] + p[3] / p[2]
            constraint_peak_val = scipy.optimize.NonlinearConstraint(peak_val, np.log(145_000), np.log(350000))
            if target is not None:
                constraint_target = scipy.optimize.NonlinearConstraint(peak_val, np.log(target - 5000), np.log(target + 5000))

            #    - contraint on the continuity of the model at the peak:
            peak_eq = lambda p: (p[5] + p[8] / p[7]) - (p[0] + p[3] / p[2])
            constraint_peak_eq = scipy.optimize.NonlinearConstraint(peak_eq, -1e-5, 1e-5)


            def derivative_around_zero(p, xs=None, h=1e-4):

                """
                Returns an array of derivative values at several x's around zero.
                The derivative is computed via a forward difference.
                """

                if xs is None:
                    # we scan from -1° to +1° in steps of 0.1°
                    xs = np.linspace(-MM_PER_DEGREE, MM_PER_DEGREE, 100)
                
                return (__model(xs + h, p) - __model(xs, p)) / h 
            

            derivative_ub = 10  # upper bound on the derivative
            xs = np.linspace(-MM_PER_DEGREE, MM_PER_DEGREE, 21)  # e.g. 21 points from -1 to +1
            # n_points = len(xs)
            # lower_bounds = -derivative_ub * np.ones(n_points)
            # upper_bounds = +derivative_ub * np.ones(n_points)

            constraint_derivative =scipy.optimize.NonlinearConstraint(
                fun=lambda p: max(abs(derivative_around_zero(p, xs=xs))),  # vector of derivatives
                lb=-derivative_ub,
                ub=derivative_ub
)
            
            

            #    - at ±3mm from the peak, the density should be at least 7200, no more than 12000:
            val_model = lambda x: lambda p: __model(x, p)
            contraint_3mm = scipy.optimize.NonlinearConstraint(val_model(3), np.log(7200), np.inf)
            contraint_m3mm = scipy.optimize.NonlinearConstraint(val_model(-3), np.log(7200), np.inf)

            lb = [  8,    0, 0.1, 0.50, c-.05,   8,    0, 0.1, 0.50] # lower bounds
            p0 = [9.4, 0.17, 0.3, 0.86, c    , 9.4, 0.17, 0.3, 0.86] # initial guess, c is the center
            ub = [ 10,  0.5, 0.6,    1, c+.05,  10,  0.5, 0.6,    1] # upper bounds
            if target is not None:
                # print("DEBUG: target:", np.log(target))
                result = scipy.optimize.minimize(
                    __loss,
                    p0,
                    args=(x_data, y_data),
                    bounds=scipy.optimize.Bounds(lb, ub),
                    constraints=[constraint_peak_val, constraint_peak_eq, 
                                contraint_3mm,
                                contraint_m3mm,
                                    constraint_derivative,
                                        constraint_target,
                                        ],
                                        options={'maxiter': 200}
                
                )
            else:
                # print("DEBUG: no target, using only constraints")
                result = scipy.optimize.minimize(
                    __loss,
                    p0,
                    args=(x_data, y_data),
                    bounds=scipy.optimize.Bounds(lb, ub),
                    constraints=[constraint_peak_val, constraint_peak_eq, 
                                contraint_3mm,
                                contraint_m3mm,
                                    constraint_derivative],
                                        options={'maxiter': 200}
                )
        # print ("DEBUG:result:", result )
            return result

        def __f_wide(f, C, mode: str):
            def __impl(f, C, x, mode: str):
                assert mode in ['upper', 'lower']
                EPS = 1e-6
                fp = lambda x: (f(x + EPS) - f(x - EPS)) / (2 * EPS)
                mode = 1 if mode == 'upper' else -1
                distance = lambda x0: np.sign(fp(x)) * mode * (x0 - x) * np.sqrt(1 + 1 / fp(x0) ** 2) - C if fp(x0) != 0 else np.inf
                try:
                    x0 = scipy.optimize.newton(distance, x if fp(x) != 0 else x + EPS)
                except RuntimeError:
                    return np.inf
                if np.sign(fp(x0)) * np.sign(fp(x)) == -1: # C is probably too large
                    return np.inf
                return f(x0) + mode * C / np.sqrt(1 + fp(x0) ** 2)
            def __f_w(x):
                if isinstance(x, (int, float)):
                    return __impl(f, C, x, mode)
                if isinstance(x, np.ndarray):
                    return np.array([__impl(f, C, xi, mode) for xi in x])
            return __f_w

        mask = densities[:, 1] > 0
        x_data = MM_PER_DEGREE * densities[mask, 0]
        y_data = np.log(densities[mask, 1])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result_c = scipy.optimize.minimize(
                lambda *args: __fit(*args).fun,
                0,
                args=(x_data, y_data),
                bounds=scipy.optimize.Bounds(-.1, .1),
                method='Powell',
                options={'maxfev': 20} #, 'disp': True}
            )
            result = __fit(result_c.x.item(), x_data, y_data)
            print("DEBUG: result:", result.x, result.fun, result.success, result.message) 

            # print("DEBUG: constraint check: peak_val", result.x[0] + result.x[3] / result.x[2], result.x[5] + result.x[8] / result.x[7])
            
            print("DEBUG: exponential peak value:", np.exp(result.x[0] + result.x[3] / result.x[2]), np.exp(result.x[5] + result.x[8] / result.x[7]))
        model_deg = lambda x: __model(x * MM_PER_DEGREE, result.x)
        true_model_deg = lambda x: np.exp(model_deg(x)) # take degree input

        W = 0.2
        lower_bound = lambda x: np.exp(__f_wide(model_deg, W, 'lower')(x))
        upper_bound = lambda x: np.exp(__f_wide(model_deg, W, 'upper')(x))

        ignored_densities = densities[
            (densities[:,1] < lower_bound(densities[:,0]))
          | (densities[:,1] > upper_bound(densities[:,0]))
        ]
        densities = densities[
            (densities[:,1] >= lower_bound(densities[:,0]))
          & (densities[:,1] <= upper_bound(densities[:,0]))
        ]

        popt = result.x.copy()
        popt[[1, 6]] *= MM_PER_DEGREE  # convert mm back to to degree
        popt[[2, 3, 4, 7, 8]] /= MM_PER_DEGREE  # convert mm back to degree
        center = popt[4]
        densities_neg = densities[(-10 <= densities[:,0] - center) & (densities[:,0] - center <   0)]
        densities_pos = densities[(  0 <= densities[:,0] - center) & (densities[:,0] - center <= 10)]
        
        # compute the width of the peak basis
        TARGET = 8000
        EPS = 1e-3
        objective = lambda x: np.abs(true_model_deg(x + EPS) - true_model_deg(x - EPS)) - 2 * EPS * TARGET
        # width defined at point for which derivative of the model is 8000 (arbitrary)
        width_neg = np.abs(root_scalar(objective, bracket=[center - 6, center - 0.5], method='bisect').root - center)
        width_pos = np.abs(root_scalar(objective, bracket=[center + 0.5, center + 6], method='bisect').root - center)

        # compute the max slope of the peak
        x_vals = np.linspace(0.001, 3, 1000)
        max_slope_neg = np.max(np.abs(np.gradient(true_model_deg(center - x_vals), x_vals)))
        max_slope_pos = np.max(np.abs(np.gradient(true_model_deg(center + x_vals), x_vals)))

        # plot the results
        density_plotter.plot_bilateral_fitting_yellott(densities_neg, densities_pos, popt[4], popt[:4], popt[5:], lower_bound, upper_bound, true_model_deg, direction, ignored_densities, right_eye=self.right_eye)

        # prepare the smoothed densities with 0.1 spacing, recentered
        densities_smthd = self.smooth_densities(densities, center)

        # prepare the fitted densities with 0.1 spacing
        eccs = np.round(np.arange(-10, 10+self.step, self.step), self.round_step)
        densities_fit = np.c_[eccs, true_model_deg(eccs + center)] # effectively recenter fitted densities
        return densities_smthd, densities_fit, true_model_deg, center, width_neg, width_pos, max_slope_neg, max_slope_pos, result.fun



    def fit_densities_alternative(self, densities: np.ndarray, density_plotter: DensityPlotter, direction: Direction) -> Tuple[np.ndarray, np.ndarray, Model, float, float, float, float, float]:

        def __model(x, p):

            """
            Piecewise double Lorentzian model with center shift and separate offsets.

            For x < p[4]:
                f(x) = p[0] / (1 + ((x - p[4]) / p[1])**2) +
                    p[2] / (1 + ((x - p[4]) / p[3])**2) + p[9]
            For x >= p[4]:
                f(x) = p[5] / (1 + ((x - p[4]) / p[6])**2) +
                    p[7] / (1 + ((x - p[4]) / p[8])**2) + p[10]
            """

            left = p[0] / (1 + ((x - p[4]) / p[1])**2) + \
                p[2] / (1 + ((x - p[4]) / p[3])**2) + p[9]

            right = p[5] / (1 + ((x - p[4]) / p[6])**2) + \
                    p[7] / (1 + ((x - p[4]) / p[8])**2) + p[10]

            return np.where(x < p[4], left, right)



        def __loss(p, x, y_data):
            """
            Assumptions:
            - data points are less reliable for smaller x
            - data points underestimates the true distribution for smaller x
            - data points may overestimate the true distribution for larger x
            """
            
            soft_l1 = lambda x, C=1: 2 * C * (np.sqrt(1 + (x / C) ** 2) - 1)
            weight_periph = 1 - np.exp(-5 * np.abs(x - p[5]))  # gives importance to non-central data
            weight_overest = 1 / (1 + np.exp(-5 * (np.abs(x - p[5]) - 1))) # =0 for central data
            weight_underest = 1 - weight_overest # =0 for peripheral data
            residuals = y_data - __model(x, p)

            C = 0.3
            loss = np.mean(weight_periph * (
                weight_overest * soft_l1(np.where(residuals <  C, residuals, 0), C)
                + weight_underest * soft_l1(np.where(residuals > -C, residuals, 0), C)
            ))
            return loss

        def __fit(c, x_data, y_data):
            if isinstance(c, np.ndarray):
                c = c.item()
            
            

            # constraints on the model:
            #    - value of the peak is constrained to be between 145_000 and 320_000:
            peak_val = lambda p: p[0] + p[3] / p[2]
            constraint_peak_val = scipy.optimize.NonlinearConstraint(peak_val, np.log(145_000), np.log(350000))
            #    - contraint on the continuity of the model at the peak:
            peak_eq = lambda p: (p[5] + p[8] / p[7]) - (p[0] + p[3] / p[2])
            constraint_peak_eq = scipy.optimize.NonlinearConstraint(peak_eq, -1e-5, 1e-5)


            def derivative_around_zero(p, xs=None, h=1e-4):

                """
                Returns an array of derivative values at several x's around zero.
                The derivative is computed via a forward difference.
                """

                if xs is None:
                    # we scan from -1° to +1° in steps of 0.1°
                    xs = np.linspace(-MM_PER_DEGREE, MM_PER_DEGREE, 100)
                
                return (__model(xs + h, p) - __model(xs, p)) / h 
            

            derivative_ub = 10  # upper bound on the derivative
            xs = np.linspace(-MM_PER_DEGREE, MM_PER_DEGREE, 21)  # e.g. 21 points from -1 to +1
            # n_points = len(xs)
            # lower_bounds = -derivative_ub * np.ones(n_points)
            # upper_bounds = +derivative_ub * np.ones(n_points)

            constraint_derivative =scipy.optimize.NonlinearConstraint(
                fun=lambda p: max(abs(derivative_around_zero(p, xs=xs))),  # vector of derivatives
                lb=-derivative_ub,
                ub=derivative_ub
)
            def ncones_in_a_circle(p, radius=1, n_radial=100, n_angular=100):

                """
                Returns the integral of the model in a circle of radius `radius` around the peak.
                The integral is computed via the trapezoidal rule.

                We try to follow the data of Zhang on the total number of cones in the fovea:
                they should remain constant for all patients at .

                Since at this step the model is fitting just one direction (x or y) depending on the call
                done later in cone density calculator, i am going to assume that the model is radially simmetric
                Big assumption, but hopefully this code does not make it in its current form
                all the way to pubblication
                If we want to keep this idea, a smarter way would be to fit both the x and y model and interpolate 
                the values of the integral in the circle (as we only have them on the x and y axis)

                Also a big mess, but slightly less big than the previous one

                """
                # 1. Define radial and angular grids
                rho_vals   = np.linspace(0, radius, n_radial)
                theta_vals = np.linspace(0, 2*np.pi, n_angular, endpoint=False)

                # 2. Compute the spacing (the sides of the pseudosquare of the grid)
                dr     = rho_vals[1] - rho_vals[0] if n_radial > 1 else radius
                dtheta = 2*np.pi / n_angular
                
                # 3. Sum up contributions
                total = 0.0
                for rho in rho_vals:
                    for theta in theta_vals:
                        density = np.exp(__model(rho, p))
                        # In polar coords, area element is (rho * dr * dtheta)
                        total += density * rho * dr * dtheta

                return total
            
            integration_radius = 1.0
            integration_ub = 1.2e5
            integration_lb = -np.inf
            constraint_area = scipy.optimize.NonlinearConstraint(
            fun=lambda p: ncones_in_a_circle(p, r=integration_radius),
            lb=integration_lb,   # lower bound
            ub=integration_ub  # upper bound
)

            #    - at ±3mm from the peak, the density should be at least 7200, no more than 12000:
            val_model = lambda x: lambda p: __model(x, p)
            contraint_3mm = scipy.optimize.NonlinearConstraint(val_model(3), np.log(7200), np.inf)
            contraint_m3mm = scipy.optimize.NonlinearConstraint(val_model(-3), np.log(7200), np.inf)

            lb = [0, 0, 0, 0, c-.05, 0, 0, 0, 0, -1,-1] # lower bounds
            p0 = [1.25, 0.25, 0.2, 0.3,  c,   1.25, 0.25, 0.2, 0.3, 0, 0]  # initial guess, c is the center
            ub = [3,  2, 1,    1, c+.05,  3,  2, 1,  1, 0.3, 0.3] # upper bounds

            result = scipy.optimize.minimize(
                __loss,
                p0,
                args=(x_data, y_data),
                bounds=scipy.optimize.Bounds(lb, ub),
                constraints=[constraint_peak_val, constraint_peak_eq, 
                             contraint_3mm,
                               contraint_m3mm,
                                #  constraint_derivative,
                                    # constraint_area
                                    ],
               
            )
            # print ("DEBUG:result:", result )
            return result

        def __f_wide(f, C, mode: str):
            def __impl(f, C, x, mode: str):
                assert mode in ['upper', 'lower']
                EPS = 1e-6
                fp = lambda x: (f(x + EPS) - f(x - EPS)) / (2 * EPS)
                mode = 1 if mode == 'upper' else -1
                distance = lambda x0: np.sign(fp(x)) * mode * (x0 - x) * np.sqrt(1 + 1 / fp(x0) ** 2) - C if fp(x0) != 0 else np.inf
                try:
                    x0 = scipy.optimize.newton(distance, x if fp(x) != 0 else x + EPS)
                except RuntimeError:
                    return np.inf
                if np.sign(fp(x0)) * np.sign(fp(x)) == -1: # C is probably too large
                    return np.inf
                return f(x0) + mode * C / np.sqrt(1 + fp(x0) ** 2)
            def __f_w(x):
                if isinstance(x, (int, float)):
                    return __impl(f, C, x, mode)
                if isinstance(x, np.ndarray):
                    return np.array([__impl(f, C, xi, mode) for xi in x])
            return __f_w

        mask = densities[:, 1] > 0
        x_data = MM_PER_DEGREE * densities[mask, 0]
        y_data = densities[mask, 1]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result_c = scipy.optimize.minimize(
                lambda *args: __fit(*args).fun,
                0,
                args=(x_data, y_data),
                bounds=scipy.optimize.Bounds(-.1, .1),
                method='Powell',
                options={'maxfev': 20}#, 'disp': True}
            )
            result = __fit(result_c.x.item(), x_data, y_data)
                
        model_deg = lambda x: __model(x * MM_PER_DEGREE, result.x)
        true_model_deg = lambda x: np.exp(model_deg(x)) # take degree input

        W = 0.2
        lower_bound = lambda x: np.exp(__f_wide(model_deg, W, 'lower')(x))
        upper_bound = lambda x: np.exp(__f_wide(model_deg, W, 'upper')(x))

        ignored_densities = densities[
            (densities[:,1] < lower_bound(densities[:,0]))
          | (densities[:,1] > upper_bound(densities[:,0]))
        ]
        densities = densities[
            (densities[:,1] >= lower_bound(densities[:,0]))
          & (densities[:,1] <= upper_bound(densities[:,0]))
        ]

        popt = result.x.copy()
        popt[[1, 6]] *= MM_PER_DEGREE  # convert mm back to to degree
        popt[[2, 3, 4, 7, 8]] /= MM_PER_DEGREE  # convert mm back to degree
        center = popt[4]
        densities_neg = densities[(-10 <= densities[:,0] - center) & (densities[:,0] - center <   0)]
        densities_pos = densities[(  0 <= densities[:,0] - center) & (densities[:,0] - center <= 10)]
        
        # compute the width of the peak basis
        TARGET = 8000
        EPS = 1e-3
        objective = lambda x: np.abs(true_model_deg(x + EPS) - true_model_deg(x - EPS)) - 2 * EPS * TARGET
        # width defined at point for which derivative of the model is 8000 (arbitrary)
        width_neg = np.abs(root_scalar(objective, bracket=[center - 6, center - 0.5], method='bisect').root - center)
        width_pos = np.abs(root_scalar(objective, bracket=[center + 0.5, center + 6], method='bisect').root - center)

        # compute the max slope of the peak
        x_vals = np.linspace(0.001, 3, 1000)
        max_slope_neg = np.max(np.abs(np.gradient(true_model_deg(center - x_vals), x_vals)))
        max_slope_pos = np.max(np.abs(np.gradient(true_model_deg(center + x_vals), x_vals)))

        # plot the results
        density_plotter.plot_bilateral_fitting_yellott(densities_neg, densities_pos, popt[4], popt[:4], popt[5:], lower_bound, upper_bound, true_model_deg, direction, ignored_densities, right_eye=self.right_eye)

        # prepare the smoothed densities with 0.1 spacing, recentered
        densities_smthd = self.smooth_densities(densities, center)

        # prepare the fitted densities with 0.1 spacing
        eccs = np.round(np.arange(-10, 10+self.step, self.step), self.round_step)
        densities_fit = np.c_[eccs, true_model_deg(eccs + center)] # effectively recenter fitted densities
        return densities_smthd, densities_fit, true_model_deg, center, width_neg, width_pos, max_slope_neg, max_slope_pos
   
    def smooth_densities(self, densities: np.ndarray, center: float) -> np.ndarray:
        """
        Smooth the densities using a lowess interpolation, filling the gaps with
        linear interpolation when the gap size is "not too large" (defined by a
        function of the eccentricity, increasing with the eccentricity). The 
        denstitites are interpolated on a grid of 0.1 spacing, using provided
        `center` as the origin of the grid.
        :param densities: The densities to smooth.
        :type densities: np.ndarray
        :return: The smoothed densities.
        :rtype: np.ndarray
        """
        lowess = np.unique(sm.nonparametric.lowess(
            densities[:,1].tolist(),
            (densities[:,0] - center).tolist(),
            frac=0.1
        ), axis=0)
        interp_func = interp1d(lowess[:, 0], lowess[:, 1], kind='linear', fill_value="extrapolate")
        CENTER_GS = 0.15
        OUTER_GS = 1
        gap_sizes = OUTER_GS - (OUTER_GS - CENTER_GS) / (1 + lowess[:,0]**2 / 10)
        gaps_indices = np.where(np.diff(lowess[:,0]) > gap_sizes[:-1])[0]
        eccs = np.round(np.arange(-10, 10+self.step, self.step), self.round_step)
        start_gaps = lowess[gaps_indices,0]
        end_gaps = lowess[gaps_indices + 1,0]
        valid_eccs_mask = ~np.any([(eccs > start) & (eccs < end) for start, end in zip(start_gaps, end_gaps)], axis=0)
        ecc_min = max(-10, self.step * np.ceil(lowess[:,0].min() / self.step))
        ecc_max = min(10, self.step * np.floor(lowess[:,0].max() / self.step) + self.step)
        valid_eccs_mask &= (eccs >= ecc_min) & (eccs <= ecc_max)
        return np.c_[eccs, np.where(valid_eccs_mask, interp_func(eccs), np.nan)]


class ConeDensityCalculator:
    """
    A class used to calculate cone densities from montage mosaic data.

    :param path_manager: The manager for processing paths.
    :type path_manager: ProcessingPathManager
    :param montage_mosaic: The montage mosaic data.
    :type montage_mosaic: MontageMosaic
    """

    def __init__(self, path_manager: ProcessingPathManager, montage_mosaic: MontageMosaic, step: float = 0.1):
        """
        Initialize the ConeDensityCalculator with the given path manager and montage mosaic.

        :param path_manager: The manager for processing paths.
        :type path_manager: ProcessingPathManager
        :param montage_mosaic: The montage mosaic data.
        :type montage_mosaic: MontageMosaic
        """
        self.step = step
        self.round_step = round(self.step * 10)
        self.path_manager = path_manager
        self.montage_mosaic = montage_mosaic
        self.atms_limit = Parser.get_density_curve_atms_limit()
        self.mdrnn_limit = Parser.get_density_curve_mdrnn_limit()
        self.total_density = Density()
        self.mdrnn_density = Density()
        self.atms_density = Density()
        self.density_plotter = DensityPlotter(self.path_manager.density.path)

    def get_densities_by_yellott(self, from_csv: bool = False, to_csv: bool = True) -> Density:
        """
        Get the cone densities using the Yellott's ring-based method. The densities are calculated
        for both the X and Y directions, and guarantees that the densities are sorted by eccentricity
        such that for X, the densities are sorted from Temporal to Nasal, and for Y, from Superior to Inferior.
        ### Warning: this method assumes that `self.montage_mosaic` has been built properly beforehand, and does not check for it.
        """
        yellott_calculator = YellottConeDensityCalculator(self.path_manager, self.montage_mosaic, self.step)
        if not from_csv:
            densities_X, densities_Y = yellott_calculator.get_densities()
        else:
            densities_X = np.loadtxt(self.path_manager.density.path / 'densities_raw_X.csv', delimiter=';', skiprows=1)
            densities_Y = np.loadtxt(self.path_manager.density.path / 'densities_raw_Y.csv', delimiter=';', skiprows=1)

        densities_smthd_X, densities_fit_X, model_X, center_X, width_temporal, width_nasal, max_slope_temporal, max_slope_nasal, resultX = yellott_calculator.fit_densities(densities_X, self.density_plotter, Direction.X)
        densities_smthd_Y, densities_fit_Y, model_Y, center_Y, width_superior, width_inferior, max_slope_superior, max_slope_inferior, resultY = yellott_calculator.fit_densities(densities_Y, self.density_plotter, Direction.Y)

        coeffX = resultY/(resultX + resultY)  # coeffX is the ratio of the Y result to the sum of the X and Y results
        coeffY = resultX/(resultX + resultY)  # coeffY is the
        print("DEBUG: coeffX:", coeffX, "coeffY:", coeffY)
        
        target = (coeffX * np.nanmax(densities_fit_X) + coeffY * np.nanmax(densities_fit_Y))  # target is the sum of the two fitted densities
        
        print ("DEBUG: target:", target)

        densities_smthd_X, densities_fit_X, model_X, center_X, width_temporal, width_nasal, max_slope_temporal, max_slope_nasal, resultX = yellott_calculator.fit_densities(densities_X, self.density_plotter, Direction.X, target=target)
        densities_smthd_Y, densities_fit_Y, model_Y, center_Y, width_superior, width_inferior, max_slope_superior, max_slope_inferior, resultY = yellott_calculator.fit_densities(densities_Y, self.density_plotter, Direction.Y, target=target)

        self.density_plotter.plot_density_curve_smoothed(densities_smthd_X, densities_smthd_Y)
        densities_avg_X = self._step_densities(densities_X, center_X)
        densities_avg_Y = self._step_densities(densities_Y, center_Y)

        if to_csv:
            # save raw densities

            pd.DataFrame(densities_X).to_csv(self.path_manager.density.path / 'densities_raw_X.csv', index=False, header=['ecc', 'dens_raw_X'], float_format='%.8g', na_rep='', sep=';')
            pd.DataFrame(densities_Y).to_csv(self.path_manager.density.path / 'densities_raw_Y.csv', index=False, header=['ecc', 'dens_raw_Y'], float_format='%.8g', na_rep='', sep=';')

            # save smoothed/fitted densities
            df = pd.DataFrame({
                'ecc': np.round(np.arange(-10, 10 + self.step, self.step), self.round_step),
                'dens_X': densities_avg_X[:,1],
                'dens_smthd_X': densities_smthd_X[:,1],
                'dens_fit_X': densities_fit_X[:,1],
                'dens_Y': densities_avg_Y[:,1],
                'dens_smthd_Y': densities_smthd_Y[:,1],
                'dens_fit_Y': densities_fit_Y[:,1]
            })
            df.loc[0, 'width_nasal'] = width_nasal
            df.loc[0, 'width_temporal'] = width_temporal
            df.loc[0, 'width_inferior'] = width_inferior
            df.loc[0, 'width_superior'] = width_superior
            df.loc[0, 'center_X'] = center_X
            df.loc[0, 'center_Y'] = center_Y
            df.loc[0, 'max_slope_nasal'] = max_slope_nasal
            df.loc[0, 'max_slope_temporal'] = max_slope_temporal
            df.loc[0, 'max_slope_inferior'] = max_slope_inferior
            df.loc[0, 'max_slope_superior'] = max_slope_superior

            print("DEBUG: saving results (called by get_densities_by yellot) in: ", self.path_manager.density.path  )
            df.to_csv(self.path_manager.density.path / 'densities.csv', index=False, header=True, float_format='%.8g', na_rep='', sep=';')

        filter_nan_out = lambda x: x[~np.isnan(x[:,1])]
        return Density(
            X=dict(filter_nan_out(densities_avg_X)),
            Y=dict(filter_nan_out(densities_avg_Y)),
            X_smoothed=dict(filter_nan_out(densities_smthd_X)),
            Y_smoothed=dict(filter_nan_out(densities_smthd_Y)),
            X_fitted=dict(densities_fit_X),
            Y_fitted=dict(densities_fit_Y)
        )
    
    def _step_densities(self, densities: np.ndarray, center: float = 0) -> np.ndarray:
        """
        Step the densities to 0.1 spacing, centered around the given `center`.
        
        :param densities: The densities to step.
        :type densities: np.ndarray
        :param center: The center of the densities.
        :type center: float
        :return: The stepped densities.
        :rtype: np.ndarray
        """
        eccs = np.round(np.arange(-10, 10 + self.step, self.step), self.round_step)
        median = lambda x: np.median(x) if len(x) > 0 else np.nan
        return np.c_[eccs, [median(densities[np.abs(densities[:,0] - center - ecc) < self.step/2][:,1]) for ecc in eccs]]

    def average_and_smoothen_sampled_densities(self, densities_mdrnn: np.ndarray, densities_atms: np.ndarray, direction: Direction) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the sampled densities from MDRNN and ATMS, weight-average them using a 
        logistic-ish function (to give more import to ATMS closer to the fovea and MDRNN further from the fovea),
        then smooth the averaged densities using LOWESS. 
        """
        densities_avg = ConeDensityCalculator._weighted_average(densities_mdrnn, densities_atms, self.mdrnn_limit, self.atms_limit)
        # lowess sometimes returns duplicate rows, so we make sure to remove them
        lowess = np.unique(sm.nonparametric.lowess(
            densities_avg[:, 1].tolist(),
            densities_avg[:, 0].tolist(),
            frac=self.step
        ), axis=0)
        interp_func = interp1d(lowess[:, 0], lowess[:, 1], kind='linear', fill_value="extrapolate")
        eccs = np.round(np.linspace(-10, 10, int(20/self.step + 1)), self.round_step)
        eccs = eccs[(min(lowess[:, 0]) <= eccs) & (eccs <= max(lowess[:, 0]))]
        densities_smthd = np.c_[eccs, interp_func(eccs)]
        
        self.density_plotter.plot_sampled_densities_axis(densities_mdrnn, densities_atms, densities_avg, densities_smthd, direction)
        return densities_avg, densities_smthd
    
    def fit_sampled_densities(
            self, 
            densities_avg_x: np.ndarray, 
            densities_avg_y: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the sampled densities to very basic exponential model. Consider using
        Yellott's method for more accurate results and models.

        :param densities_avg_x: The averaged densities in the X direction.
        :type densities_avg_x: np.ndarray
        :param densities_avg_y: The averaged densities in the Y direction.
        :type densities_avg_y: np.ndarray
        :param exp_model: The exponential model to fit the densities to. The model should take an array of eccentricities [°] and an array of 3 parameters as input, and return the fitted densities [cell/mm²].
        :type exp_model: Callable[[np.ndarray, np.ndarray], np.ndarray]
        :return: A tuple containing the fitted densities in the X direction, the fitted densities in the Y direction, the model parameters for the X direction, and the model parameters for the Y direction.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        def prepare_densities(densities: np.ndarray) -> np.ndarray:
            densities[:, 0] = np.abs(densities[:, 0])
            densities = densities[(1 <= densities[:, 0]) & (densities[:, 0] <= 10)]
            # ignore points for which density is so low that is an obvious mistake
            densities = densities[densities[:,1] > 20000 / (1 + densities[:,0] ** 1.4)]
            densities = np.array(sorted(densities, key=lambda p: p[0]))
            return densities

        def fit_curve(densities: np.ndarray, 
                    model: Callable, 
                    p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            residuals_fun = lambda p, x, y: model(x, p) - y
            results = least_squares(residuals_fun, p0, args=(densities[:,0], densities[:,1]), loss='soft_l1', f_scale=3000, bounds=(0, np.inf))
            return results.x, results.fun

        densities_avg_x = prepare_densities(densities_avg_x)
        densities_avg_y = prepare_densities(densities_avg_y)

        exp_model = lambda x, p: p[0] + p[1] * np.exp(- p[2] * x)

        p0 = np.array([6000, 90000, 0.5])

        popt_x, residuals_x = fit_curve(densities_avg_x, exp_model, p0)
        popt_y, residuals_y = fit_curve(densities_avg_y, exp_model, p0)

        eccs = np.round(np.linspace(-10, 10, 201), self.round_step)
        densities_fit_x = np.c_[eccs, exp_model(eccs, popt_x)]
        densities_fit_y = np.c_[eccs, exp_model(eccs, popt_y)]

        self.density_plotter.plot_density_curve_fitting(densities_avg_x, densities_avg_y, popt_x, popt_y, residuals_x, residuals_y, exp_model)
        return densities_fit_x, densities_fit_y, popt_x, popt_y

    @staticmethod
    def _weighted_average(densities_mdrnn: np.ndarray, densities_atms: np.ndarray, mdrnn_limit: float, atms_limit: float) -> np.ndarray:
        """
        Compute the weighted average of the densities, using a logistic-ish function to weight the densities.

        :param densities_mdrnn: The densities from MDRNN sorted by eccentricity, eccentricities in first column, densities in second columns.
        :type densities_mdrnn: np.ndarray
        :param densities_atms: The densities from ATMS sorted by eccentricity, eccentricities in first column, densities in second columns.
        :type densities_atms: np.ndarray
        :param mdrnn_limit: The eccentricity limit for MDRNN densities (ignore MDRNN densities with eccentricity < mdrnn_limit).
        :type mdrnn_limit: float
        :param atms_limit: The eccentricity limit for ATMS densities (ignore ATMS densities with eccentricity > atms_limit).
        :type atms_limit: float
        :return: The weighted average of the densities.
        :rtype: np.ndarray
        """
        def logi(x : np.ndarray) -> np.ndarray:
            EPS = 0.05
            result = np.zeros_like(x)
            mask = (x > EPS) & (x < 1 - EPS)
            result[mask] = 1 / (1 + np.exp((1 - 2*x[mask]) / (x[mask] - x[mask]**2)))
            result[x >= 1 - EPS] = 1                
            return result

        atms_ticks = densities_atms[:,0].copy()
        weights = logi((np.abs(atms_ticks) - mdrnn_limit) / (atms_limit - mdrnn_limit))

        densities_mdrnn_on_atms_ticks = np.interp(atms_ticks, densities_mdrnn[:,0], densities_mdrnn[:,1], left=0, right=0)
        
        avg = np.c_[atms_ticks, (1 - weights) * densities_atms[:,1] + weights * densities_mdrnn_on_atms_ticks]
        min_ecc, max_ecc = np.min(atms_ticks), np.max(atms_ticks)
        return np.r_[densities_mdrnn[densities_mdrnn[:,0] < min_ecc], avg, densities_mdrnn[densities_mdrnn[:,0] > max_ecc]] 

    @deprecated(reason='Use get_densities_by_yellott or get_densities_by_sampling instead.')
    def get_densities(self) -> Tuple[Density, Density, Density]:
        """
        **[LEGACY] method to get the densities for the cones. Probably breaks. 
        This does not guarantee that densities are sorted by eccentricity 
        Temporal to Nasal (X) and Superior to Inferior (Y), and is more sensitive
        to bad image quality than the new methods.**

        This method gathers cones, plots initial cones, calculates counts, plots cones with distance,
        merges cone locations, plots merged cones, calculates locations, and calculates area per locations.
        Finally, it processes the locations and returns the densities.

        :return: A tuple containing total density, mdrnn density, and atms density.
        :rtype: Tuple[Density, Density, Density]
        """
        cones_mdrnn, cones_atms = self._gather_cones()
        cone_mosaic_plotter = ConeMosaicPlotter(self.montage_mosaic, self.path_manager.path / 'density_analysis')

        self._plot_initial_cones(cone_mosaic_plotter, cones_mdrnn, cones_atms)

        counts_calculator = ConeNumberPerDistanceCalculator(self.montage_mosaic.transformed_center)
        self._calculate_counts(counts_calculator, cones_mdrnn, cones_atms)

        self._plot_cones_with_distance(cone_mosaic_plotter)

        cones_location_per_distance_atms = self._merge_cone_locations(self.cones_location_per_distance_x_atms, self.cones_location_per_distance_y_atms)
        cones_location_per_distance_mdrnn = self._merge_cone_locations(self.cones_location_per_distance_x_mdrnn, self.cones_location_per_distance_y_mdrnn)

        self._plot_merged_cones(cone_mosaic_plotter, cones_location_per_distance_atms, cones_location_per_distance_mdrnn)
        self._calculate_locations()
        self._calculate_area_per_locations()

        return self.process_locations()

    def _plot_initial_cones(self, plotter: ConeMosaicPlotter, cones_mdrnn: List[List[int]], cones_atms: List[List[int]]) -> None:
        """
        Plot the initial cones.

        :param plotter: The plotter for cone mosaics.
        :type plotter: ConeMosaicPlotter
        :param cones_mdrnn: The list of mdrnn cones.
        :type cones_mdrnn: List[List[int]]
        :param cones_atms: The list of atms cones.
        :type cones_atms: List[List[int]]
        """
        plotter.plot_cones(cones_mdrnn, name='mdrnn')
        plotter.plot_cones(cones_atms, name='atms')
        plotter.plot_all_cones(cones_mdrnn, cones_atms)

    def _calculate_counts(
        self,
        calculator: ConeNumberPerDistanceCalculator,
        cones_mdrnn: List[List[int]],
        cones_atms: List[List[int]]) -> None:
        """
        Calculate the counts of cones per distance.

        :param calculator: The calculator for cone numbers per distance.
        :type calculator: ConeNumberPerDistanceCalculator
        :param cones_mdrnn: The list of mdrnn cones.
        :type cones_mdrnn: List[List[int]]
        :param cones_atms: The list of atms cones.
        :type cones_atms: List[List[int]]
        """
        self.counts_x_mdrnn, self.counts_y_mdrnn, self.cones_location_per_distance_x_mdrnn, self.cones_location_per_distance_y_mdrnn = calculator.get_counts(cones_mdrnn, self.montage_mosaic.binary_map(self.path_manager.path))
        self.counts_x_atms, self.counts_y_atms, self.cones_location_per_distance_x_atms, self.cones_location_per_distance_y_atms = calculator.get_counts(cones_atms, self.montage_mosaic.binary_map(self.path_manager.path))

    def _plot_cones_with_distance(self, plotter: ConeMosaicPlotter) -> None:
        """
        Plot the cones with distance.

        :param plotter: The plotter for cone mosaics.
        :type plotter: ConeMosaicPlotter
        """
        plotter.plot_with_distance(self.cones_location_per_distance_x_mdrnn, 'mdrnn_X')
        plotter.plot_with_distance(self.cones_location_per_distance_y_mdrnn, 'mdrnn_Y')
        plotter.plot_with_distance(self.cones_location_per_distance_x_atms, 'atms_X')
        plotter.plot_with_distance(self.cones_location_per_distance_y_atms, 'atms_Y')

    def _plot_cones_with_rois(self, plotter: ConeMosaicPlotter, solved_mdrnn_cones: List[List[int]], solved_atms_cones: List[List[int]], path: Path | None = None) -> None:
        plotter.plot_cones_with_rois(solved_mdrnn_cones, solved_atms_cones, path)

    def _merge_cone_locations(self, cones_x: Dict[str, List[int]], cones_y: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        Merge cone locations from x and y directions.

        :param cones_x: The cone locations in the x direction.
        :type cones_x: Dict[str, List[int]]
        :param cones_y: The cone locations in the y direction.
        :type cones_y: Dict[str, List[int]]
        :return: The merged cone locations.
        :rtype: Dict[str, List[int]]
        """
        merged_cones = cones_x
        for key, value_list in cones_y.items():
            merged_cones[key].extend(value_list)
        return merged_cones

    def _plot_merged_cones(self, plotter: ConeMosaicPlotter, cones_atms: List[List[int]], cones_mdrnn: List[List[int]]) -> None:
        """
        Plot the merged cones.

        :param plotter: The plotter for cone mosaics.
        :type plotter: ConeMosaicPlotter
        :param cones_atms: The list of atms cones.
        :type cones_atms: List[List[int]]
        :param cones_mdrnn: The list of mdrnn cones.
        :type cones_mdrnn: List[List[int]]
        """
        plotter.plot_with_distance(cones_atms, 'atms')
        plotter.plot_with_distance(cones_mdrnn, 'mdrnn')

    def _calculate_locations(self) -> None:
        """
        Calculate the locations for the cones.

        This method calculates the locations based on the counts of cones in x and y directions.
        """
        all_keys = set(self.counts_x_mdrnn) | set(self.counts_y_mdrnn) | set(self.counts_x_atms) | set(self.counts_y_atms)
        min_location, max_location = min(all_keys, default=0), max(all_keys, default=0)
        adjusted_min, adjusted_max = min_location - self.step / 2 if min_location < 0 else min_location, max_location + self.step / 2
        self.locations = PIXELS_PER_DEGREE * np.arange(adjusted_min, adjusted_max, step=self.step)

    def _calculate_area_per_locations(self) -> None:
        """
        Calculate the area per location for both x and y directions.

        This method initializes the area_per_locations_x and area_per_locations_y attributes
        by using the RefactoredPixelPerDistanceCalculator to get the counts of pixels per distance unit
        in the x and y directions, respectively.
        """
        center = Point(self.montage_mosaic.transformed_center.x, self.montage_mosaic.transformed_center.y)
        self.area_per_locations_x = RefactoredPixelPerDistanceCalculator(min(self.locations), max(self.locations), center).get_counts(self.montage_mosaic.binary_map(self.path_manager.path), 'x')
        self.area_per_locations_y = RefactoredPixelPerDistanceCalculator(min(self.locations), max(self.locations), center).get_counts(self.montage_mosaic.binary_map(), 'y')

    def process_locations(self) -> Tuple[Density, Density, Density]:
        """
        Process the locations to calculate densities for both x and y directions.

        This method calculates the densities for both x and y directions and plots the results.
        It returns the total density, mdrnn density, and atms density.

        :return: A tuple containing total density, mdrnn density, and atms density.
        :rtype: Tuple[Density, Density, Density]
        """
        for direction in ['x', 'y']:
            self.calculate_density_for_direction(direction)
        DensityPlotter(self.path_manager.path / 'density_analysis').plot(self.total_density, self.mdrnn_density, self.atms_density)
        return self.total_density, self.mdrnn_density, self.atms_density

    def calculate_density_for_direction(self, direction: str):
        """
        Calculate the density for a given direction.

        This method calculates the density for the specified direction ('x' or 'y') and updates
        the corresponding density attributes.

        :param direction: The direction to calculate density for ('x' or 'y').
        :type direction: str
        """
        if direction == 'x':
            counts_mdrnn, counts_atms = self.counts_x_mdrnn, self.counts_x_atms
            area_per_locations = self.area_per_locations_x
        else:
            counts_mdrnn, counts_atms = self.counts_y_mdrnn, self.counts_y_atms
            area_per_locations = self.area_per_locations_y

        for i in range(len(self.locations) - 1):
            a, b = self.locations[i], self.locations[i + 1]
            distance = round((a + b) / (2 * PIXELS_PER_DEGREE), self.round_step)
            if (distance not in area_per_locations) & (counts_mdrnn[distance] > 0):
                if distance > 0:
                    counts_mdrnn[round(distance - self.step, self.round_step)] += counts_mdrnn[distance]
                    counts_mdrnn[distance] = 0
                else:
                    counts_mdrnn[round(distance + self.step, self.round_step)] += counts_mdrnn[distance]
                    counts_mdrnn[distance] = 0
                continue
            elif (distance not in area_per_locations) & (counts_atms[distance] > 0):
                print("no")
                continue
            elif distance not in area_per_locations:
                continue
            elif area_per_locations[distance] <= AREA_THRESHOLD:
                continue
            average_density = None
            if abs(distance) > self.mdrnn_limit:
                density_mdrnn = self.compute_density(counts_mdrnn[distance], area_per_locations[distance])
                self.mdrnn_density.add_density(direction, distance, density_mdrnn)
                average_density = density_mdrnn
            if abs(distance) < self.atms_limit:
                density_atms = self.compute_density(counts_atms[distance], area_per_locations[distance])
                self.atms_density.add_density(direction, distance, density_atms)
                if average_density is None:
                    average_density = density_atms
                else:
                    average_density = (average_density + density_atms) / 2
            self.total_density.add_density(direction, distance, average_density)

    def compute_density(self, counts: int, area: int) -> float:
        """
        Compute the density given the counts and area.

        This method calculates the density based on the counts and area, considering the
        PIXELS_PER_DEGREE and MM_PER_DEGREE constants.

        :param counts: The number of counts.
        :type counts: int
        :param area: The area value.
        :type area: int
        :return: The computed density.
        :rtype: float
        """
        return (PIXELS_PER_DEGREE / MM_PER_DEGREE) ** 2 * counts / area if area > AREA_THRESHOLD else 0

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

if __name__ == "__main__":
    from src.cell.montage.montage_mosaic_builder import CorrectedMontageMosaicBuilder
    path_manager: ProcessingPathManager = ProcessingPathManager(Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject101\Session470'))
    path_manager.montage.initialize_montage()
    builder = CorrectedMontageMosaicBuilder(path_manager) # CorrectedMontageMosaicBuilder(self.path_manager.montaged_path)
    mosaic = builder.build_mosaic()
    # # mosaic.show()
    import pickle
    # with open(r"C:\Users\BardetJ\Downloads\mosaic.pkl", "wb") as f:
    #     pickle.dump(mosaic, f)
    # with open(r"C:\Users\BardetJ\Downloads\mosaic.pkl", "rb") as f:
    #     mosaic = pickle.load(f)
    results = ConeDensityCalculator(path_manager, mosaic).get_densities()
    print("finished")