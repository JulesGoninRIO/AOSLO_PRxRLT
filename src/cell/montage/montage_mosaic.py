from typing import Dict, Union, List, Tuple
import re
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pickle
from shapely.geometry import box, Polygon, Point, LineString
from shapely.affinity import affine_transform

from src.shared.computer_vision.point import Point as CVPoint
from src.shared.datafile.image_file import ImageFile
from src.shared.datafile.coordinates_file import CoordinatesFile
from src.cell.affine_transform import AffineTransform
from src.cell.processing_path_manager import ProcessingPathManager
from src.cell.montage.montage_element import MontageElement, transform_elements, get_corners, correct_transforms,get_element_neighbors

class MontageMosaic:
    """
    Class for creating and managing a montage mosaic.

    This class handles the creation, transformation, and management of montage elements
    to build a mosaic image.

    :param path_manager: The path manager for handling file paths.
    :type path_manager: ProcessingPathManager
    :param has_been_corrected: Flag indicating if the mosaic has been corrected.
    :type has_been_corrected: bool
    :param center_point: The center point of the mosaic.
    :type center_point: Point
    :param mosaic_shape: The shape of the mosaic.
    :type mosaic_shape: Tuple[int, int]
    :param ratio: The ratio for scaling the mosaic.
    :type ratio: Tuple[float, float]
    :param fundus_shape: The shape of the fundus.
    :type fundus_shape: Tuple[int, int]
    """
    def __init__(self, path_manager: ProcessingPathManager, has_been_corrected: bool = False, center_point: Point | CVPoint | None = None, mosaic_shape: Tuple[int, int] | None = None, ratio: Tuple[float, float] = (1, 1), fundus_shape: Tuple[int, int] | None = None):
        """
        Initialize the MontageMosaic.

        :param path_manager: The path manager for handling file paths.
        :type path_manager: ProcessingPathManager
        :param has_been_corrected: Flag indicating if the mosaic has been corrected.
        :type has_been_corrected: bool
        :param center_point: The center point of the mosaic.
        :type center_point: Point
        :param mosaic_shape: The shape of the mosaic.
        :type mosaic_shape: Tuple[int, int]
        :param ratio: The ratio for scaling the mosaic.
        :type ratio: Tuple[float, float]
        :param fundus_shape: The shape of the fundus.
        :type fundus_shape: Tuple[int, int]
        """
        self.path_manager = path_manager
        self.montage_elements : List[MontageElement] = []
        self.has_been_corrected = has_been_corrected
        self.center = center_point
        self.mosaic_shape = mosaic_shape
        self.ratio = ratio
        self.fundus_shape = fundus_shape
        if center_point:
            self.transformed_center = CVPoint(self.center.x * ratio[0], self.center.y * ratio[1])
        # self.overlap_regions = {}
        self.overlap_regions = []
        self._mosaic_cache = None
        self._binary_map_cache = None

    def set_transforms(self):
        """
        Set the transformations for all montage elements.

        This method applies the scaling ratio to all montage elements.
        """
        for element in self.montage_elements:
            element.set_transform(self.ratio)

    def get_shape(self) -> Tuple[int, int]:
        """
        Get the shape of the mosaic.

        This method returns the shape of the mosaic, either from the provided mosaic shape
        or by calculating the corners of the montage elements.

        :return: The shape of the mosaic.
        :rtype: Tuple[int, int]
        """
        if self.mosaic_shape:
            return self.mosaic_shape
        else:
            corners = get_corners(self.montage_elements, self.ratio, self.path_manager.path)
            return (corners[0], corners[1])

    def add_element(self, element: MontageElement):
        """
        Add a montage element to the mosaic.

        :param element: The montage element to add.
        :type element: MontageElement
        """
        self.montage_elements.append(element)

    def get_element(self, element_name: Union[str, ImageFile]) -> MontageElement:
        """
        Get a montage element by name or ImageFile.

        This method retrieves a montage element based on the provided name or ImageFile.

        :param element_name: The name or ImageFile of the element to retrieve.
        :type element_name: Union[str, ImageFile]
        :return: The corresponding montage element.
        :rtype: MontageElement
        :raises ValueError: If no element with the given name is found.
        """
        for element in self.montage_elements:
            if isinstance(element_name, ImageFile):
                if element.image_file.is_same_except_modality(element_name):
                    return element
            else:
                try:
                    if element.image_file.is_same_except_modality(ImageFile(element_name)):
                        return element
                except ValueError:
                    if element.image_file.is_same_except_modality(CoordinatesFile(element_name).to_image_file()):
                        return element
        raise ValueError(f"No element with name {element_name} found")

    def get_transform(self, element_name: Union[str, ImageFile]) -> AffineTransform:
        """
        Get the transformation for a montage element.

        This method retrieves the transformation for the specified montage element.
        If the transformation is not set, it sets the transformations for all elements first.

        :param element_name: The name or ImageFile of the element to retrieve the transformation for.
        :type element_name: Union[str, ImageFile]
        :return: The transformation of the specified element.
        :rtype: AffineTransform
        """
        transform = self.get_element(element_name).transform
        if transform is None:
            self.set_transforms()
            transform = self.get_element(element_name).transform
        return transform

    def get_element_neighbors_with_distance(self, element: MontageElement) -> Dict[ImageFile, int]:
        """
        Get neighbors of a montage element with their distances.

        This method retrieves the neighbors of the given montage element and calculates the distance
        to each neighbor. If the montage has not been corrected, non-overlapping patches are not treated as neighbors.

        :param element: The montage element to find neighbors for.
        :type element: MontageElement
        :return: A dictionary with neighbors' image files as keys and their distances as values.
        :rtype: Dict[ImageFile, int]
        """
        neighbors = get_element_neighbors(element, self.montage_elements)
        neighbors_with_distance = {}

        for neighbor in neighbors:
            # If the montage has not been corrected, we assume non-overlapping
            # patch distance to be unknown thus we do not treat them as neighbors
            if not self.has_been_corrected:
                if element.transform.compute_overlap_region(neighbor.transform, self.ratio).area <= 1e-10:
                    continue
            distance = element.transform.translation_distance(neighbor.transform, self.ratio)
            neighbors_with_distance[neighbor.image_file] = distance

        return neighbors_with_distance

    def get_overlap_between_neighbors(self, element: MontageElement) -> List[Union[Polygon, Point, LineString]]:
        """
        Get overlap regions between a montage element and its neighbors.

        This method calculates the overlap regions between the given montage element and its neighbors.

        :param element: The montage element to find overlaps for.
        :type element: MontageElement
        :return: A list of overlap regions.
        :rtype: List[Union[Polygon, Point, LineString]]
        """
        neighbors_image_names = get_element_neighbors(element, self.montage_elements)
        overlaps = []
        for neighbor in neighbors_image_names:
            overlap = element.transform.compute_overlap_region(neighbor.transform) #, self.ratio)
            if overlap.is_empty:
                continue
            overlaps.append(overlap)
        return overlaps

    def set_overlap_regions(self):
        """
        Set the overlap regions for all montage elements.

        This method calculates and sets the overlap regions for all montage elements in the mosaic.
        """
        for montage_element in self.montage_elements:
            self.overlap_regions.extend(self.get_overlap_between_neighbors(montage_element))

    def draw_overlap_regions_on_images(self):
        """
        Draw overlap regions on images.

        This method reads the image data for each montage element, draws the overlap regions on the images,
        and displays them.

        """
        for montage_element in self.montage_elements:
            montage_element.image_file.read_data(self.path_manager.path)
            image = montage_element.image_file.data
            overlap_regions = self.overlap_regions[str(montage_element.image_file)]
            for neighbor_name, overlap_region in overlap_regions.items():
                if overlap_region.area > 1e-10:
                    overlap_region = detransform(overlap_region, montage_element.transform.matrix, self.ratio)
                    x, y = overlap_region.exterior.xy
                    plt.plot(x, y, color='red')
            plt.imshow(image)
            plt.show()

    def find_montage_element_by_image_file(self, image_file: ImageFile) -> MontageElement:
        """
        Find a montage element by its image file.

        This method searches for a montage element that matches the given image file.

        :param image_file: The image file to search for.
        :type image_file: ImageFile
        :return: The montage element that matches the image file, or None if not found.
        :rtype: MontageElement
        """
        for element in self.montage_elements:
            if element.image_file == image_file:
                return element
        return None

    def _prepare_mosaic(self, ratio: Tuple[int, int], new_shape: bool = False) -> np.array:
        """
        Prepare the mosaic array.

        This method prepares the mosaic array based on the given ratio and shape parameters.

        :param ratio: The ratio for scaling the mosaic.
        :type ratio: Tuple[int, int]
        :param new_shape: Flag indicating whether to create a new shape for the mosaic.
        :type new_shape: bool
        :return: The prepared mosaic array.
        :rtype: np.array
        """
        if self.mosaic_shape:
            mosaic = np.zeros(self.mosaic_shape)
        elif new_shape:
            corners = get_corners(self.montage_elements, ratio, self.path_manager.path)
            mosaic = np.zeros((corners[0], corners[1]))
        elif self._mosaic_cache is not None:
            mosaic = np.zeros(self._mosaic_cache.shape)
        elif self._binary_map_cache is not None:
            mosaic = np.zeros(self._binary_map_cache.shape)
        else:
            corners = get_corners(self.montage_elements, ratio, self.path_manager.path)
            correct_transforms(self.montage_elements, corners)
            mosaic = np.zeros((corners[1]-corners[3], corners[0]-corners[2]))
            self.transformed_center = CVPoint(self.transformed_center.x - corners[3], self.transformed_center.y - corners[2])
        return mosaic

    def get_mosaic(self, path: Path = None) -> np.array:
        """
        Get the mosaic image.

        This method retrieves the mosaic image, either from the cache or by creating it.

        :param path: The path to read image data from, if needed.
        :type path: Path
        :return: The mosaic image array.
        :rtype: np.array
        """
        if self._mosaic_cache is not None:
            return self._mosaic_cache

        for element in self.montage_elements:
            if len(element.image_file.data) == 0:
                element.image_file.read_data(path)

        mosaic = self._create_mosaic(lambda element: element.image_file.data, self.ratio)
        self._mosaic_cache = mosaic
        return mosaic

    def _create_mini_mosaic(self, get_data_func: callable) -> np.array:
        """
        Create a mini mosaic.

        This method creates a mini mosaic using the provided data function.

        :param get_data_func: The function to get data for each montage element.
        :type get_data_func: callable
        :return: The mini mosaic array.
        :rtype: np.array
        """
        ratio = (1, 1)
        mosaic = self._prepare_mosaic(ratio, new_shape=True)
        for i, element in enumerate(self.montage_elements):
            data = get_data_func(element)
            try:
                transform = element.fundus_transform.matrix[:2]
                transformed_data = cv2.warpAffine(data, transform, (mosaic.shape[1], mosaic.shape[0]))
                mosaic += transformed_data
            except cv2.error:
                transformed_data = cv2.warpAffine(np.uint8(data * 255), element.transform.matrix[:2], (mosaic.shape[1], mosaic.shape[0]))
                if i == 0:
                    mosaic = mosaic > 0
                mosaic += transformed_data > 0
        return mosaic

    def _create_mosaic(self, get_data_func: callable, ratio: Tuple[int, int]=None) -> np.array:
        """
        Create the mosaic.

        This method creates the mosaic using the provided data function and ratio.

        :param get_data_func: The function to get data for each montage element.
        :type get_data_func: callable
        :param ratio: The ratio for scaling the mosaic, defaults to None.
        :type ratio: Tuple[int, int], optional
        :return: The mosaic array.
        :rtype: np.array
        """
        if ratio is None:
            ratio = self.ratio if self.ratio is not None else (1, 1)
        mosaic = self._prepare_mosaic(ratio)
        for i, element in enumerate(self.montage_elements):
            data = get_data_func(element)
            try:
                transformed_data = cv2.warpAffine(data, element.transform.matrix[:2], (mosaic.shape[1], mosaic.shape[0]))
                try:
                    mosaic += transformed_data
                except ValueError:
                    try:
                        mosaic += transformed_data
                    except ValueError:
                        try:
                            mosaic += np.transpose(transformed_data)
                        except ValueError:
                            transformed_data_shape = transformed_data.shape
                            mosaic_shape = mosaic.shape
                            pad_height = mosaic_shape[0] - transformed_data_shape[0]
                            pad_width = mosaic_shape[1] - transformed_data_shape[1]
                            padded_transformed_data = np.pad(transformed_data,
                                                            ((0, pad_height), (0, pad_width)),
                                                            mode='constant',
                                                            constant_values=0)
                            mosaic += padded_transformed_data
            except cv2.error:
                transformed_data = cv2.warpAffine(np.uint8(data * 255), element.transform.matrix[:2], (mosaic.shape[1], mosaic.shape[0]))
                if i == 0:
                    mosaic = mosaic > 0
                try:
                    mosaic += transformed_data > 0
                except ValueError:
                    mosaic += np.transpose(transformed_data > 0)
        return mosaic

    def binary_map(self, path: Path = None) -> np.array:
        """
        Generate a binary map of the montage.

        This method generates a binary map of the montage, either from the cache or by creating it.

        :param path: The path to read image data from, if needed.
        :type path: Path
        :return: The binary map array.
        :rtype: np.array
        """
        if self._binary_map_cache is not None:
            return self._binary_map_cache

        for element in self.montage_elements:
            if len(element.image_file.data) == 0:
                element.image_file.read_data(path)

        binary_map = self._create_mosaic(lambda element: element.image_file.dark_regions)
        self._binary_map_cache = binary_map
        return binary_map

    def save(self):
        """
        Save the mosaic image.

        This method saves the mosaic image to the corrected path.
        """
        if not self.path_manager.montage.corrected_path.exists():
            os.makedirs(self.path_manager.montage.corrected_path)
        mosaic = self.get_mosaic(self.path_manager.path)
        cv2.imwrite(str(self.path_manager.montage.corrected_path / 'mosaic.tif'), mosaic.astype(np.uint8))
        cv2.imwrite(str(self.path_manager.montage.corrected_path / 'mosaic_flipped.tif'), cv2.flip(mosaic, 1).astype(np.uint8))
        with open(str(self.path_manager.montage.corrected_path / 'mosaic.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def save_on_fundus(self, fundus: np.array):
        """
        Save the mini mosaic on the fundus image.

        This method saves the mini mosaic overlaid on the fundus image to the corrected path.

        :param fundus: The fundus image array.
        :type fundus: np.array
        """
        if not self.path_manager.montage.corrected_path.exists():
            os.makedirs(self.path_manager.montage.corrected_path)
        mini_mosaic = self._create_mini_mosaic(lambda element: element.image_file.data)
        fundus = cv2.flip(fundus, 1)
        resized_fundus = cv2.resize(fundus, self.fundus_shape)
        pad_height = resized_fundus.shape[0] - mini_mosaic.shape[0]
        pad_width = resized_fundus.shape[1] - mini_mosaic.shape[1]
        padded_mini_mosaic = np.pad(
            mini_mosaic, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0
        )

        combined_image = cv2.add(padded_mini_mosaic.astype(np.uint8), resized_fundus.astype(np.uint8))
        cv2.imwrite(str(self.path_manager.montage.corrected_path / 'mosaic_fundus.tif'), combined_image.astype(np.uint8))
        # TODO: save also the flipped one
        cv2.imwrite(str(self.path_manager.montage.corrected_path / 'mosaic_fundus_flipped.tif'), cv2.flip(combined_image, 1).astype(np.uint8))

    def save_images(self):
        """
        Save individual images of the montage elements.

        This method saves individual images of the montage elements and their corresponding calculated split and OA850nm images.
        """
        if self.mosaic_shape is None:
            mosaic = self._prepare_mosaic(self.ratio)
            self.mosaic_shape = mosaic.shape
        for element in self.montage_elements:
            self.save_image(element)
            cs_file = ImageFile(re.sub('Confocal', 'CalculatedSplit', str(element.image_file)))
            cs_file.read_data(self.path_manager.path)
            cs_element = MontageElement(cs_file, transform=element.transform)
            self.save_image(cs_element)
            oa_file = ImageFile(re.sub('Confocal', 'OA850nm', str(element.image_file)))
            try:
                oa_file.read_data(self.path_manager.path)
            except ValueError:
                continue
            oa_element = MontageElement(oa_file, transform=element.transform)
            self.save_image(oa_element)

    def save_image(self, element: MontageElement):
        """
        Save an individual image of a montage element.

        This method saves an individual image of the given montage element to the corrected path.

        :param element: The montage element to save.
        :type element: MontageElement
        """
        if not self.path_manager.montage.corrected_path.exists():
            os.makedirs(self.path_manager.montage.corrected_path)
        transformed_data = cv2.warpAffine(element.image_file.data, element.transform.matrix[:2], (self.mosaic_shape[1], self.mosaic_shape[0]))
        cv2.imwrite(str(self.path_manager.montage.corrected_path / str(element.image_file)), transformed_data.astype(np.uint8))

def detransform(polygon: Polygon, transform: np.ndarray, ratio: Tuple[int, int]) -> Polygon:
    """
    Detransform a polygon using the given transformation matrix and ratio.

    This method applies the inverse affine transformation to the given polygon.

    :param polygon: The polygon to be detransformed.
    :type polygon: Polygon
    :param transform: The transformation matrix.
    :type transform: np.ndarray
    :param ratio: The ratio for scaling the transformation.
    :type ratio: Tuple[int, int]
    :return: The detransformed polygon.
    :rtype: Polygon
    """
    transform[0, :] *= ratio[0]
    transform[1, :] *= ratio[0]
    translate_x = transform[0, 2]
    translate_y = transform[1, 2]
    inverse_translation_param = [1, 0, 0, 1, -translate_x, -translate_y]

    transformed_polygon = affine_transform(polygon, inverse_translation_param)

    return transformed_polygon


    #TODO: get out of the montage mosaic, also only allows the montage mosaic to be built from image that are not crops
    # def get_patch_neighbor_with_distance(self, element: MontageElement, neighbors: List[MontageElement]):
    #     neighbors_with_distance = {}

    #     patch_transfom = element.transform.matrix - np.array([
    #         [1, 0, element.image_file.crop_x_position],
    #         [0, 1, element.image_file.crop_y_position]
    #     ])

    #     for neighbor in neighbors:
    #         neighbor_transform = neighbor.transform.matrix - np.array([
    #             [1, 0, neighbor.image_file.crop_x_position],
    #             [0, 1, neighbor.image_file.crop_y_position]
    #         ])
    #         distance = AffineTransform.from_csv(patch_transfom).transform.compute_distance(neighbor_transform)
    #         neighbors_with_distance[str(neighbor.image_file)] = distance

    #     return neighbors_with_distance


#TODO: keep if needed
# def find_rightmost_transform(affine_transforms):
#     # Assume affine_transforms is a list of 3x3 numpy arrays
#     max_x_translation = -np.inf
#     rightmost_transform = None
#     for transform in affine_transforms:
#         x_translation = transform[0, 2]
#         if x_translation > max_x_translation:
#             max_x_translation = x_translation
#             rightmost_transform = transform
#     return rightmost_transform


#TODO: this works with result from MATLAB MONTAGE only so will be include in Matlab class or module
# Also to avoid having ot compute the shifts each time would be good to save a pickle object or else
# representing the shifts of a montage -> easier and faster to load (cause also need in results)