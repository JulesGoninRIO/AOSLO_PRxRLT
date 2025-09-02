import warnings
from typing import Union

import numpy as np
import scipy
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numba import njit
from scipy import spatial
from typing import List, Dict, Tuple, Callable
import logging
from pathlib import Path

from src.cell.cell_detection.atms._local_maxima import get_photoreceptor_centers
from src.cell.cell_detection.atms._arrays import concatenate_arrays_with_unique_elements#, get_array_statistics
from src.cell.cell_detection.atms._constants import NON_EXISTENT_VERTEX, GOOD_NUMBER_OF_NEIGHBORS, \
    PIXEL_COORDINATES_DATATYPE, EMPTY_PIXEL_COORDINATES, GlobalThresholdContainer, \
    GlobalParameterContainer
from scipy.spatial import ConvexHull
# from .global_helpers import get_array_statistics

@njit(cache=True)
def get_array_statistics(array: np.ndarray, n_bins: int = 100) -> Tuple[float, float]:
    """
    Get the array statistics

    :param array: array of any size and shape with elements of any numerical type
    :type array: np.ndarray
    :param n_bins: umber bins in which the number of elements will be count;
        the larger it is, the better will be the resolution, but less robust will be the answer, defaults to 100
    :type n_bins: int, optional
    :return: the most probable value of the array (according to element number in bins) and its standard deviation

    :rtype: Tuple[float, float]
    """

    # here we do not perform type and value check of the inputs, because it does not work under njit() decorator
    if array.size == 0:
        return 0, 0
    elif array.size == 1:
        return array.flatten()[0], 0

    histogram, bin_edges = np.histogram(array, n_bins)

    smooth_window_size = 5
    histogram = np.convolve(histogram, np.ones(smooth_window_size) / smooth_window_size)
    maxima_idx = np.argsort(histogram)[-2:]

    # if we have two maximums with close values, take an average of them to be more robust
    if histogram[maxima_idx[1]] > histogram[maxima_idx[0]] * 2:
        mu = float(bin_edges[maxima_idx[1]])
    else:
        # be careful not to index values out of the array
        mu = (bin_edges[min(maxima_idx)] + bin_edges[min(max(maxima_idx) + 1, bin_edges.size - 1)]) / 2
    # shift the mean value of the array to our mu and compute std then
    std = float(np.std(array - np.mean(array) + mu))
    return mu, std

GOOD_CELL_THRESHOLD = 0.05

@njit(cache=True)
def _sigma(x: Union[float, np.ndarray], threshold: float) -> np.ndarray:
    """
    Compute the sigmoid function.

    This function computes the sigmoid of the input value(s) `x` with a given `threshold`.

    :param x: The input value(s) which can be a float or a numpy array.
    :type x: Union[float, np.ndarray]
    :param threshold: The threshold value for the sigmoid function.
    :type threshold: float
    :return: The computed sigmoid value(s).
    :rtype: np.ndarray
    """
    return 1. / (1. + np.exp(threshold - x))

class VoronoiDiagram:
    """
    Envelope of the scipy.spatial.Voronoi class that introduces additional
    useful methods and allows the usage of the same iterator for the regions
    and points.
    """

    region_colors = {
        GOOD_NUMBER_OF_NEIGHBORS - 2: 'blue',
        GOOD_NUMBER_OF_NEIGHBORS - 1: 'violet',
        GOOD_NUMBER_OF_NEIGHBORS: 'green',
        GOOD_NUMBER_OF_NEIGHBORS + 1: 'yellow',
        GOOD_NUMBER_OF_NEIGHBORS + 2: 'red'
    }
    """
    Defines which color will a region have based on the number of its neighbors.
    """

    alpha = 0.1
    """The transparency of the region color on the diagram."""

    def __init__(self, points: np.ndarray, image: np.ndarray = None) -> None:
        """
        Build the Voronoi diagram of the cells and prepare the neighbor
        information for further use.

        :param points: (N, 2) array of (y, x) point coordinates in pixels on the image.
        :type points: np.ndarray
        :param image: Grayscale image for which the points have been detected.
        :type image: np.ndarray, optional
        :raise RuntimeError: If too few points were provided to construct the diagram.
        """
        self.points = points
        try:
            self._diagram = spatial.Voronoi(points, incremental=True)
        except (scipy.spatial.QhullError, IndexError, ValueError):
            logging.info('Too few cells provided for initialization')
            raise RuntimeError('Too few cells provided for initialization')

        if image is not None:
            self.quality_image = image
            self._point_qualities = self.quality_image[self.point_coordinates[:, 0], self.point_coordinates[:, 1]]

        self._distances_between_neighbors = np.empty(self.number_of_regions, dtype=np.float32)
        self._determine_region_neighbors()

        self.number_of_neighbors = np.empty(self.number_of_regions, dtype=np.uint8)
        for i in range(self.number_of_regions):
            self.number_of_neighbors[i] = np.count_nonzero(
                self._diagram.ridge_points[:, 0] == i
                ) + np.count_nonzero(
                    self._diagram.ridge_points[:, 1] == i
                    )
        self.number_of_good_cells: int = np.count_nonzero(
            self.number_of_neighbors == GOOD_NUMBER_OF_NEIGHBORS
        )
        self.number_of_bad_cells: int = self.number_of_regions - self.number_of_good_cells

    @property
    def number_of_regions(self) -> int:
        """
        Get the number of points in the diagram.

        :return: Number of points in the diagram.
        :rtype: int
        """
        return self._diagram.points.shape[0]

    @property
    def point_coordinates(self) -> np.ndarray:
        """
        Get the coordinates of all the points in the diagram.

        :return: (N, 2) array of integer (y, x) coordinates of all the points in pixels.
        :rtype: np.ndarray
        """
        return np.asarray(self._diagram.points, dtype=PIXEL_COORDINATES_DATATYPE)

    def get_neighbor_idx(self, point_ind: int) -> np.array:
        """
        Get the indices of the neighbors of the given point.

        :param point_ind: Index of the point.
        :type point_ind: int
        :return: (K, 2) array of indices of the neighbors of the given point.
        :rtype: np.array
        """
        return np.concatenate(
            (
                self._diagram.ridge_points[self._diagram.ridge_points[:, 0] == point_ind, 1],
                self._diagram.ridge_points[self._diagram.ridge_points[:, 1] == point_ind, 0]
            )
        )

    def get_distances_to_neighbors(self, point_ind: int) -> np.ndarray:
        """
        Get the distances to the neighbors of the given point.

        :param point_ind: Index of the point.
        :type point_ind: int
        :return: K-length array of distances to the neighbors of the given point.
        :rtype: np.ndarray
        """
        return np.concatenate(
            (
                self._distances_between_neighbors[self._diagram.ridge_points[:, 0] == point_ind],
                self._distances_between_neighbors[self._diagram.ridge_points[:, 1] == point_ind]
            )
        )
    def _colorize_regions(self, ax: Axes) -> None:
        """
        Plot on the given axes a polygon for each of the Voronoi regions.
        Fill these polygons with the color determined by the number of points' neighbors.

        :param ax: axes of the matplotlib plot where the regions will be drawn
        """

        vert_coords = self._diagram.vertices
        for i in range(self.number_of_regions):

            region_ind = self._diagram.point_region[i]
            vert_idx = self._diagram.regions[region_ind]

            x_coords = vert_coords[vert_idx, 0]
            y_coords = vert_coords[vert_idx, 1]

            neighb_num = len(vert_idx)
            if NON_EXISTENT_VERTEX not in vert_idx:
                if neighb_num <= GOOD_NUMBER_OF_NEIGHBORS - 2:
                    ax.fill(x_coords, y_coords, VoronoiDiagram.region_colors[GOOD_NUMBER_OF_NEIGHBORS - 2],
                            alpha=VoronoiDiagram.alpha)
                elif neighb_num >= GOOD_NUMBER_OF_NEIGHBORS + 2:
                    ax.fill(x_coords, y_coords, VoronoiDiagram.region_colors[GOOD_NUMBER_OF_NEIGHBORS + 2],
                            alpha=VoronoiDiagram.alpha)
                else:
                    ax.fill(x_coords, y_coords, VoronoiDiagram.region_colors[neighb_num], alpha=VoronoiDiagram.alpha)

    def display_on_image(self, title: str = '') -> None:
        """
        Draw the colorized Voronoi regions on top of the image for which they were constructed.
        Though the point coordinates are in (y, x) format with the origin in the top left corner,
        the built-in voronoi plot considers them to be in (x, y) format with the origin in the bottom left corner,
        so as it is, the regions will be plotted incorrectly.
        To handle this, we rotate the image (we could just switch the columns in the coordinates array instead,
        but it would require creation of a copy of them and would make the usage of a region colorizer more complex).
        """
        fig, ax = plt.subplots()

        # diagram has the origin at the left bottom corner and the first coordinate is horizontal,
        # so we transform the image correspondingly
        image = np.flipud(np.rot90(self.quality_image))
        ax.imshow(image, cmap='gray')

        self._colorize_regions(ax)
        spatial.voronoi_plot_2d(self._diagram,
                                ax,
                                point_size=3,
                                line_colors='yellow',
                                line_alpha=0.05,
                                show_vertices=False)
        if title:
            plt.title(title)
        plt.show()

    def _determine_region_neighbors(self) -> None:
        """
        Constructs the connectivity list for the points of the diagram.
        It is a M-length array of float denoting the distances between neighbor pairs written in the
        self._diagram.ridge_points array.
        """

        # ridge_points contains the point indices between each ridge lies
        idx1 = self._diagram.ridge_points[:, 0]
        idx2 = self._diagram.ridge_points[:, 1]
        # we use the diagram._points directly instead of  to omit type conversions
        coord1 = self._diagram.points[idx1]
        coord2 = self._diagram.points[idx2]
        self._distances_between_neighbors = np.sqrt(np.sum((coord1 - coord2) ** 2, axis=1))

    def gather_statistics_on_neighbor_distances(self) -> np.ndarray:
        """
        Here we only take into account the cells with GOOD_NUMBER_OF_NEIGHBORS neighbors.

        :return: array of distances between all pairs of neighbors, at least one of which has exactly
            GOOD_NUMBER_OF_NEIGHBORS neighbors
        """

        distances = np.empty(0, dtype=np.float32)
        for cell_ind in range(self.number_of_regions):
            if GOOD_NUMBER_OF_NEIGHBORS - 1 <= self.number_of_neighbors[cell_ind] <= GOOD_NUMBER_OF_NEIGHBORS + 1:
                neighbor_distances = self.get_distances_to_neighbors(cell_ind)
                distances = np.concatenate((distances, neighbor_distances))
        return distances

        # distances = np.empty(self.number_of_good_cells * GOOD_NUMBER_OF_NEIGHBORS, dtype=np.float32)
        # # for all cells with good number of neighbors
        # good_cells_idx = np.where(self.number_of_neighbors == GOOD_NUMBER_OF_NEIGHBORS)[0]
        # for i, good_cell_ind in enumerate(good_cells_idx):
        #     distances_to_neighbors = self.get_distances_to_neighbors(good_cell_ind)
        #     distances[GOOD_NUMBER_OF_NEIGHBORS * i: GOOD_NUMBER_OF_NEIGHBORS * (i + 1)] = \
        #         distances_to_neighbors[:]
        # return distances

    def assess_neighbor_distance_metrics(self) -> np.ndarray:
        """
        After obtaining the array of distances between neighbors, determines the local threshold
        (as a combination of mean and std of this array). Based of this local threshold, the global one is updated,
        being the average of the last few local ones.
        Then determines the distance to the closest neighbor for each cell and takes it as the metric.

        :return: the sigmoid of the metrics centered at the global threshold or an all-zeros array,
            if the diagram is too irregular (it means that the cells were detected badly)
        :raise RuntimeError: if the metrics cannot be determined or it is too different from what we expect
        """

        if self.number_of_good_cells <= self.number_of_regions * GOOD_CELL_THRESHOLD:
            raise RuntimeError('There are no good cells')

        # determine the good values for the distances
        distances = self.gather_statistics_on_neighbor_distances()
        mu, std = get_array_statistics(distances)

        # remove outliers and recompute the statistics
        distances = distances[np.logical_and(distances < mu + 2 * std, distances > mu - 2 * std)]
        mu, std = get_array_statistics(distances)
        local_threshold = mu - 2 * std

        # if the variance is too high, then there are too few cells detected,
        # so we will mark them all as invalid to get rid of this ROI
        if local_threshold < 0:
            raise RuntimeError('The mean value of the distance is too low.')

        # the same if the distances are too big
        global_thresholds = GlobalThresholdContainer()
        if global_thresholds.removing_distance_threshold is not None \
                and local_threshold > global_thresholds.removing_distance_threshold * 5:
            raise RuntimeError('The mean value of the distance is too high.')

        # update the threshold as an exponential average
        global_thresholds.removing_distance_threshold = local_threshold

        # compute the metrics for each cell
        metrics = np.empty(self.number_of_regions)
        for i in range(metrics.size):
            # base our metrics on the distance to the closest neighbor
            metrics[i] = np.min(self.get_distances_to_neighbors(i))

        return _sigma(metrics, global_thresholds.removing_distance_threshold)

    def mark_valid_cells(self) -> np.ndarray:
        """
        Determines two metrics for each point. The first one is the luminosity of the image below,
        and the second one is based on the distance to its neighbors. Marks the points with bad values of the metrics
        as invalid (which means they they are actually not cells, but false positives of the detection).
        If the

        :return: boolean mask with True values for good points
        """

        lum_mu, lum_std = get_array_statistics(self._point_qualities)
        global_parameters = GlobalParameterContainer()
        lum_thr = lum_mu - global_parameters.refinement_removing_luminosity_std_multiplier * lum_std
        global_thresholds = GlobalThresholdContainer()
        global_thresholds.removing_luminance_threshold = lum_thr

        # initial estimation of which cells are valid is made with only the luminance criterion
        validity_mask = np.where(self._point_qualities > global_thresholds.removing_luminance_threshold, True, False)

        try:
            neighbor_distance_criterion = self.assess_neighbor_distance_metrics()
        except RuntimeError as e:
            validity_mask.fill(False)
            # warnings.warn(e.args[0], UserWarning)
            logging.info(e.args[0])
            return validity_mask

        low_distance_mask = neighbor_distance_criterion < 0.5

        # cells with good luminance and bad distance need to be checked
        # we cannot simply remove them all, because from each group of close cells we should leave at least one
        candidates = np.where(np.logical_and(validity_mask, low_distance_mask))[0]
        if candidates.size == 0:
            return validity_mask

        # sort candidates by their qualities (the best ones first)
        candidate_qualities = self._point_qualities[candidates]
        idx = np.flip(np.argsort(candidate_qualities))
        candidates = candidates[idx]

        # we select a non-deleted cell with next best quality and mark all its lower-quality neighbors as deleted
        for i in range(candidates.size - 1):
            if not validity_mask[candidates[i]]:
                continue
            for j in range(i + 1, candidates.size):
                if candidates[j] in self.get_neighbor_idx(candidates[i]):
                    validity_mask[candidates[j]] = False
        return validity_mask

    def predict_new_cells(self) -> np.ndarray:
        """
        Based on the distance metrics determined for the good cells, it determines the local threshold on it
        and updates the global one. Based on it, the too far neighbors are determined, and between them is performed
        a search for the missed cells.

        :return: the (N, 2) array of (y, x) coordinates of both the existing cells and the new ones, or an empty array,
            if the diagram turned to be too irregular.
        """

        if self.number_of_good_cells <= self.number_of_regions * GOOD_CELL_THRESHOLD:
            warnings.warn('There are no good cells for prediction', UserWarning)
            return EMPTY_PIXEL_COORDINATES

        # determine the range of distances between cells when they are considered far
        dist_mu, dist_std = get_array_statistics(self.gather_statistics_on_neighbor_distances())
        dist_thr = dist_mu

        # # these are the cases when we have too few cells left after the removing
        if dist_mu < dist_std:
            # warnings.warn('Distance deviation is too high', UserWarning)
            logging.info('Distance deviation is too high')
            # self.display_on_image()
            return EMPTY_PIXEL_COORDINATES
        local_dist_thr = dist_mu - dist_std

        global_thresholds = GlobalThresholdContainer()

        # if the distances between cells are too large, it means that there are too few of them detected,
        # and the region is in fact bad
        if global_thresholds.addition_distance_threshold is not None \
                and local_dist_thr > 5 * global_thresholds.addition_distance_threshold:
            # warnings.warn('Local threshold for distance on prediction is too large', UserWarning)
            logging.info('Local threshold for distance on prediction is too large')
            return EMPTY_PIXEL_COORDINATES

        global_thresholds.addition_distance_threshold = local_dist_thr

        lum_mu, lum_std = get_array_statistics(self._point_qualities)
        global_parameters = GlobalParameterContainer()
        # lum_thr = lum_mu - (global_parameters.refinement_luminosity_std_multiplier - 1.5) * lum_std
        lum_thr = lum_mu - global_parameters.refinement_addition_luminosity_std_multiplier * lum_std

        updated_cell_coordinates = self.point_coordinates
        recheck_mask = np.zeros_like(self.quality_image, dtype=bool)

        # go over all neighbor pairs and mark the area between the too far ones for rechecking
        for i, dist in np.ndenumerate(self._distances_between_neighbors):
            if dist > dist_thr:
                ind1, ind2 = self._diagram.ridge_points[i]
                coord1 = self.point_coordinates[ind1]
                coord2 = self.point_coordinates[ind2]

                up = min(coord1[0], coord2[0])
                bottom = max(coord1[0], coord2[0]) + 1
                left = min(coord1[1], coord2[1])
                right = max(coord1[1], coord2[1]) + 1
                recheck_mask[up: bottom, left: right] = True

        # among the pixels marked for rechecking we search for local maxima with good luminosity
        new_neighbor_coordinates = get_photoreceptor_centers(self.quality_image, mask=recheck_mask)
        new_neighbor_luminosities = self.quality_image[new_neighbor_coordinates[:, 0], new_neighbor_coordinates[:, 1]]
        new_neighbor_coordinates = new_neighbor_coordinates[new_neighbor_luminosities > lum_thr, :]

        updated_cell_coordinates = concatenate_arrays_with_unique_elements(
            updated_cell_coordinates,
            new_neighbor_coordinates,
            global_thresholds.addition_distance_threshold
        )

        return updated_cell_coordinates

    def remove_points(self, validity_mask: np.ndarray) -> None:
        """
        Reinitializes the object using only the points marked with True values in the given mask.

        :raise ValueError: if less than 10 points remain
        """

        valid_coordinates = self.point_coordinates[validity_mask]

        if valid_coordinates.size > 10:
            self.__init__(valid_coordinates, self.quality_image)
        else:
            raise ValueError('Almost all points have been removed')

    def draw_voronoi(
        self,
        image_name: str,
        binary_map: np.ndarray,
        overlap_region: Dict[str, np.ndarray],
        area_to_analyze_image: Dict[str, float],
        draw_dir: str) -> None:
        """
        Draws a Voronoi diagram between the cone centers in the image

        :param image_name: the name of the image
        :type image_name: str
        :param binary_map: the map with the cones
        :type binary_map: np.ndarray
        :param overlap_region: the overlap regioin between images
        :type overlap_region: Dict[str, np.ndarray]
        :param area_to_analyze_image: the area to analyze in the images
        :type area_to_analyze_image: Dict[str, float]
        :param draw_dir: the directory to output the drawn image
        :type draw_dir: str
        """

        # let's prepare the object with which the Voronoi will be created
        rect = (0, 0, self.quality_image.shape[0], self.quality_image.shape[1])
        subdiv = cv2.Subdiv2D(rect)
        # Insert points into subdiv
        for cone in self.points :
            subdiv.insert([int(cone[1]), int(cone[0])])
        (facets, self.points) = subdiv.getVoronoiFacetList([])

        # validate centers and facets that must lie within the image to be drawn
        correct_facets = []
        correct_centers = []
        for i in range(len(facets)) :
            facet = facets[i]
            all_good = True
            for single_facet in facet:
                if not 0<=round(single_facet[0])<self.quality_image.shape[0] or not \
                    0<=round(single_facet[1])<self.quality_image.shape[1]:
                    all_good = False
                elif not binary_map[round(single_facet[1]), round(single_facet[0])]:
                    all_good = False
                elif image_name in overlap_region.keys():
                    if (np.array([round(single_facet[1]), round(single_facet[0])]) \
                        == overlap_region[image_name]).all(axis = 1).any():
                        all_good = False
            if all_good :
                correct_facets.append(facet)
                correct_centers.append(self.points[i])

        #draw the correct centers and Voronoi edges
        for i in range(len(correct_facets)) :
            area =  ConvexHull(correct_facets[i]).area
            area_to_analyze = 0.6*area
            # add center and rayon to dictionnary
            area_to_analyze_image[str((correct_centers[i][0], correct_centers[i][1]))] \
                = np.sqrt(area_to_analyze/np.pi)

            facet = np.array(correct_facets[i], int)
            ifacet_arr = []
            for f in facet :
                ifacet_arr.append(f)
            ifacet = np.array(ifacet_arr, int)
            ifacets = np.array([ifacet])
            # Draw polylines in red
            cv2.polylines(self.quality_image, ifacets, True, (0, 0, 255), 0, cv2.LINE_AA, 0)

        # Draw the radius of the cones that correspond to 60% of the cone area
        for center, r in area_to_analyze_image.items():
            center = eval(center)
            self.quality_image = cv2.circle(self.quality_image,
                                            (int(center[0]), int(center[1])),
                                            radius=round(r), color=(0, 255, 0),
                                            thickness=1)
            out_name =  os.path.splitext(image_name)[0] + "_area.tif"
            cv2.imwrite(os.path.join(draw_dir, out_name), self.quality_image)

    def draw(self, image_name: str, out_path: Path):
        """
        Draw the Voronoi diagram on the image and save it.

        This method prepares the Voronoi diagram from the points and draws the facets
        and circles representing 60% of the cone area on the image. The resulting image
        is saved to the specified output path.

        :param image_name: The name of the image file.
        :type image_name: str
        :param out_path: The path where the output image will be saved.
        :type out_path: Path
        """
        # let's prepare the object with which the Voronoi will be created
        rect = (0, 0, self.quality_image.shape[1], self.quality_image.shape[0])
        subdiv = cv2.Subdiv2D(rect)
        # Insert points into subdiv
        for cone in self.points:
            try:
                subdiv.insert([int(cone[1]), int(cone[0])])
            except cv2.error:
                logging.info('Point is outside the image')
        (facets, self.points) = subdiv.getVoronoiFacetList([])

        area_to_analyze_image = {}

        # Draw the correct centers and Voronoi edges
        for i in range(len(facets)):
            area = ConvexHull(facets[i]).area
            area_to_analyze = 0.6 * area
            # Add center and radius to dictionary
            area_to_analyze_image[str((self.points[i][0], self.points[i][1]))] = np.sqrt(area_to_analyze / np.pi)

            facet = np.array(facets[i], int)
            ifacet_arr = []
            for f in facet:
                ifacet_arr.append(f)
            ifacet = np.array(ifacet_arr, int)
            ifacets = np.array([ifacet])
            # Draw polylines in red
            cv2.polylines(self.quality_image, ifacets, True, (0, 0, 255), 0, cv2.LINE_AA, 0)

        # Draw the radius of the cones that correspond to 60% of the cone area
        for center, r in area_to_analyze_image.items():
            center = eval(center)
            self.quality_image = cv2.circle(self.quality_image,
                                            (int(center[0]), int(center[1])),
                                            radius=round(r), color=(0, 255, 0),
                                            thickness=1)
        out_name = os.path.splitext(image_name)[0] + "_area.tif"
        cv2.imwrite(str(out_path / out_name), self.quality_image)

    def get_radius(self, binary_map: np.ndarray) -> Dict[str, float]:
        """
        Get the radius of the cones that correspond to 60% of the cone area.

        This method prepares the Voronoi diagram from the points and calculates the radius
        of the cones that correspond to 60% of the cone area. Only the facets and centers
        that lie within the image and the binary map are considered.

        :param binary_map: A binary map indicating valid regions.
        :type binary_map: np.ndarray
        :return: A dictionary with the center coordinates as keys and the radius as values.
        :rtype: Dict[str, float]
        """
        area_to_analyze_image = {}
        # let's prepare the object with which the Voronoi will be created
        rect = (0, 0, self.quality_image.shape[1], self.quality_image.shape[0])
        subdiv = cv2.Subdiv2D(rect)
        # Insert points into subdiv
        for cone in self.points:
            try:
                subdiv.insert([int(cone[1]), int(cone[0])])
            except cv2.error:
                logging.info('Point is outside the image')
        (facets, self.points) = subdiv.getVoronoiFacetList([])

        # Validate centers and facets that must lie within the image to be drawn
        correct_facets = []
        correct_centers = []
        for i in range(len(facets)):
            facet = facets[i]
            all_good = True
            for single_facet in facet:
                if not 0 <= round(single_facet[0]) < self.quality_image.shape[1] or not \
                        0 <= round(single_facet[1]) < self.quality_image.shape[0]:
                    all_good = False
                elif not binary_map[round(single_facet[1]), round(single_facet[0])]:
                    all_good = False
            if all_good:
                correct_facets.append(facet)
                correct_centers.append(self.points[i])

        # Draw the correct centers and Voronoi edges
        for i in range(len(correct_facets)):
            area = ConvexHull(correct_facets[i]).area
            area_to_analyze = 0.6 * area
            # Add center and radius to dictionary
            area_to_analyze_image[str((correct_centers[i][0], correct_centers[i][1]))] = np.sqrt(area_to_analyze / np.pi)
        return area_to_analyze_image
