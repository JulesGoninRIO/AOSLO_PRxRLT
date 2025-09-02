from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import math
from pathlib import Path

from src.cell.affine_transform import AffineTransform
from src.cell.montage.constants import IMAGE_SIZE
from src.shared.datafile.image_file import ImageFile

class RegionOfInterest:
    """
    A base class representing a region of interest in an image.

    :param reference_point: The reference point coordinates of the region
    :type reference_point: np.ndarray
    :param density: Optional density value associated with the region
    :type density: float | None
    """

    def __init__(self, reference_point: np.ndarray, density: Optional[float] = None) -> None:
        self.reference_point = reference_point.astype(int)
        self.density: Optional[float] = density
    
    def transform(self, transform: Optional[AffineTransform]) -> None:
        """
        Apply an affine transformation to the region's reference point.

        :param transform: The affine transformation to apply
        :type transform: AffineTransform | None
        """
        if transform:
            self.reference_point = transform.transform_points(self.reference_point).squeeze().astype(int)

    def draw(self, img_array: np.ndarray, color: Union[List[int], Tuple[int, int, int]] = [0, 0, 255]) -> None:
        """
        Draw the region of interest on the image.

        :param img_array: The image array to draw on
        :type img_array: np.ndarray
        :param color: RGB color values for drawing, defaults to [0, 0, 255]
        :type color: List[int] | Tuple[int, int, int]
        """
        pass

class DiskROI(RegionOfInterest):
    """
    A circular region of interest in an image.

    Inherits from RegionOfInterest and represents a circular area defined by a center point and radius.

    :param disk_center: The center coordinates of the disk
    :type disk_center: np.ndarray
    :param radius: The radius of the disk
    :type radius: int | float
    :param density: Optional density value associated with the region
    :type density: float | None
    """

    def __init__(self, disk_center: np.ndarray, radius: Union[int, float], density: Optional[float] = None) -> None:
        super().__init__(disk_center, density)
        self.radius = radius

    def draw(self, img_array: np.ndarray, color: Union[List[int], Tuple[int, int, int]] = [0, 0, 255]) -> None:
        """
        Draw the disk region of interest on the image.

        Draws both the circle outline and its center point.

        :param img_array: The image array to draw on
        :type img_array: np.ndarray
        :param color: RGB color values for drawing, defaults to [0, 0, 255]
        :type color: List[int] | Tuple[int, int, int]
        """
        cv2.circle(img_array, tuple(self.reference_point), radius=self.radius, color=color)
        cv2.circle(img_array, tuple(self.reference_point), radius=1, color=color)

    def __str__(self) -> str:
        """
        Return a string representation of the DiskROI.

        :return: String description of the disk ROI
        :rtype: str
        """
        return f"DiskRIO: {self.reference_point}, {self.radius}, density: {self.density}"


class MontageElement:
    """
    A class to represent a montage element, encapsulating both ImageFile and its spatial information.

    :param image_file: The image file associated with the montage element.
    :type image_file: ImageFile
    :param path: The path to the image file.
    :type path: Path, optional
    :param transform: The affine transform applied to the image file.
    :type transform: AffineTransform, optional
    :param fundus_transform: The fundus transform applied to the image file.
    :type fundus_transform: AffineTransform, optional
    """
    # Composite pattern to encapsulate both ImageFile and its spatial information
    # Created to keep ImageFile lightweight (not knowing its spatial info) and
    # keep MontageElement less complex
    corners = np.array([[0, 0, 1], [0, IMAGE_SIZE, 1], [IMAGE_SIZE, 0, 1], [IMAGE_SIZE, IMAGE_SIZE, 1]])

    def __init__(self, image_file: ImageFile, path: Path | None = None, transform: AffineTransform | None = None, fundus_transform: AffineTransform | None = None):
        self.image_file = image_file
        self.transform = transform
        self.fundus_transform = fundus_transform
        self.transformed_data = None
        self.path = path
        self.roi_atms: List[RegionOfInterest] = []
        self.roi_mdrnn: List[RegionOfInterest] = []

    def __str__(self) -> str:
        """
        Return the string representation of the montage element.

        :return: The string representation of the image file.
        :rtype: str
        """
        return str(self.image_file)

    def set_transform(self, ratio: Tuple[float, float]):
        """
        Set the transform for the montage element based on the given ratio.

        :param ratio: The ratio to apply to the transform.
        :type ratio: Tuple[float, float]
        """
        transform = self.fundus_transform.matrix.copy()
        transform[0, 0] *= ratio[0]
        transform[0, 1] *= ratio[0]
        transform[1, 0] *= ratio[1]
        transform[1, 1] *= ratio[1]
        # transform[0, 1] *= ratio[0]
        # transform[1, 0] *= ratio[1]
        transform[0, 2] *= ratio[0]
        transform[1, 2] *= ratio[1]
        self.transform = AffineTransform.from_cv2(transform)

    def get_transformed_data(self):
        """
        Get the transformed data of the image file. 

        This method applies the affine transform to the image file data and stores the result.

        :return: The transformed image data.
        :rtype: np.ndarray
        """
        if self.transformed_data is None:
            transform = self.transform.matrix
            transformed_corners = np.dot(transform, self.corners.T).T
            self.max_x = np.max(transformed_corners[:, 0])
            self.max_y = np.max(transformed_corners[:, 1])
            self.transformed_data = cv2.warpAffine(self.image_file.data, transform[:2], (round(self.max_x), round(self.max_y)))

    def get_ratio_transformed_data(self, ratio: Tuple[float, float], corners: Tuple[int, int]):
        """
        Get the transformed data of the image file based on the given ratio and corners.

        :param ratio: The ratio to apply to the transform.
        :type ratio: Tuple[float, float]
        :param corners: The corners of the transformed image.
        :type corners: Tuple[int, int]
        :return: The transformed image data.
        :rtype: np.ndarray
        """
        if self.transformed_data is None:
            from src.cell.affine_transform import adapt_transform_for_new_size
            # transform = adapt_transform_for_new_size(self.transform.matrix, ratio)
            # corners = np.array([
            #     [corners[2], corners[3], 1],
            #     [corners[2], corners[1], 1],
            #     [corners[0], corners[3], 1],
            #     [corners[0], corners[1], 1]])
            # transformed_corners = np.dot(transform, np.array(corners).T).T
            # self.max_x = np.max(transformed_corners[:, 0])
            # self.max_y = np.max(transformed_corners[:, 1])
            self.transformed_data = cv2.warpAffine(self.image_file.data, self.transform.matrix[:2], (corners[0], corners[1]))

    def get_mask(self):
        """
        Get the mask of the transformed image file.

        This method applies the affine transform to a mask of ones with the same shape as the image file data.

        :return: The transformed mask.
        :rtype: np.ndarray
        """
        if self.transformed_data is None:
            transform = self.transform.matrix
            transformed_corners = np.dot(transform, self.corners.T).T
            self.max_x = np.max(transformed_corners[:, 0])
            self.max_y = np.max(transformed_corners[:, 1])
            self.transformed_mask = cv2.warpAffine(np.ones(self.image_file.data.shape), transform[:2], (self.max_x, self.max_y))
        else:
            self.transformed_mask = cv2.warpAffine(np.ones(self.image_file.data.shape), self.transform.matrix[:2], (int(self.max_x), int(self.max_y)))

    def get_mask_data(self, mask: np.ndarray, corners: Tuple[int, int]):
        """
        Get the mask data of the transformed image file based on the given mask and corners.

        :param mask: The mask to apply the transform to.
        :type mask: np.ndarray
        :param corners: The corners of the transformed mask.
        :type corners: Tuple[int, int]
        :return: The transformed mask data.
        :rtype: np.ndarray
        """
        self.transformed_mask = cv2.warpAffine(np.ones(mask.shape), self.transform.matrix[:2], (corners[0], corners[1]))

    def build_neighbors(self) -> List[Tuple[int, int]]:
        """
        Build the list of neighboring elements based on the current element's position.

        :return: A list of tuples representing the neighboring positions.
        :rtype: List[Tuple[int, int]]
        """
        location = [self.image_file.x_position, self.image_file.y_position]
        neighbors = [(location[0]+1, location[1]),
                    (location[0]-1, location[1]),
                    (location[0], location[1]+1),
                    (location[0], location[1]-1)]
                    # (location[0]-1, location[1]-1),
                    # (location[0]-1, location[1]+1),
                    # (location[0]+1, location[1]-1),
                    # (location[0]+1, location[1]+1)]
        return neighbors

def transform_elements(elements: List[MontageElement], ratio: float, path: Path) -> List[np.ndarray]:
    """
    Transform the elements based on the given ratio and path.

    This method sets the transform for each element, reads the image data if not already read,
    and calculates the transformed corners and bounding box. It then applies the affine transform
    to each element's image data.

    :param elements: The list of montage elements to transform.
    :type elements: List[MontageElement]
    :param ratio: The ratio to apply to the transform.
    :type ratio: float
    :param path: The path to the image files.
    :type path: Path
    :return: A list of transformed image data arrays and the bounding box corners.
    :rtype: List[np.ndarray], Tuple[int, int, int, int]
    """
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    corners = np.array([[0, 0, 1], [0, IMAGE_SIZE, 1], [IMAGE_SIZE, 0, 1], [IMAGE_SIZE, IMAGE_SIZE, 1]])
    for element in elements:
        element.set_transform(ratio)
        if element.image_file.data.size == 0:
            element.image_file.read_data(path)
        transform = element.transform.matrix.copy()
        transformed_corners = np.dot(transform, corners.T).T
        min_x = min(min_x, np.min(transformed_corners[:, 0]))
        min_y = min(min_y, np.min(transformed_corners[:, 1]))
        max_x = max(max_x, np.max(transformed_corners[:, 0]))
        max_y = max(max_y, np.max(transformed_corners[:, 1]))
    corners = (math.ceil(max_x), math.ceil(max_y), math.floor(min_x), math.floor(min_y))
    return [cv2.warpAffine(
        element.image_file.data, element.transform.matrix[:2],
        (corners[0], corners[1])
        ) for element in elements], corners

def get_corners(elements: List[MontageElement], ratio: float, path: Path) -> Tuple[int, int, int, int]:
    """
    Get the bounding box corners for the transformed elements.

    This method calculates the transformed corners and bounding box for each element
    based on the given ratio and path.

    :param elements: The list of montage elements to transform.
    :type elements: List[MontageElement]
    :param ratio: The ratio to apply to the transform.
    :type ratio: float
    :param path: The path to the image files.
    :type path: Path
    :return: The bounding box corners.
    :rtype: Tuple[int, int, int, int]
    """
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    corners = np.array([[0, 0, 1], [0, IMAGE_SIZE, 1], [IMAGE_SIZE, 0, 1], [IMAGE_SIZE, IMAGE_SIZE, 1]])
    for element in elements:
        if element.image_file.data.size == 0:
            element.image_file.read_data(path)
        if ratio == (1,1):
            transform = element.fundus_transform.matrix.copy()
        else:
            if not element.transform:
                element.set_transform(ratio)
            transform = element.transform.matrix.copy()
        transformed_corners = np.dot(transform, corners.T).T
        min_x = min(min_x, np.min(transformed_corners[:, 0]))
        min_y = min(min_y, np.min(transformed_corners[:, 1]))
        max_x = max(max_x, np.max(transformed_corners[:, 0]))
        max_y = max(max_y, np.max(transformed_corners[:, 1]))
    corners = (math.ceil(max_x), math.ceil(max_y), math.floor(min_x), math.floor(min_y))
    return corners

def correct_transforms(elements: List[MontageElement], corners: Tuple[int, int, int, int]):
    """
    Correct the transforms for the elements based on the given corners.

    This method adjusts the translation component of each element's transform matrix
    based on the bounding box corners.

    :param elements: The list of montage elements to correct.
    :type elements: List[MontageElement]
    :param corners: The bounding box corners.
    :type corners: Tuple[int, int, int, int]
    """
    for element in elements:
        element.transform.matrix[0, 2] -= corners[2]
        element.transform.matrix[1, 2] -= corners[3]


def get_element_neighbors(element: MontageElement, other_elements: List[MontageElement]) -> List[MontageElement]:
    """
    Get the neighboring elements of a given montage element.

    This method finds and returns the neighboring elements of the given montage element
    from a list of other montage elements.

    :param element: The montage element for which to find neighbors.
    :type element: MontageElement
    :param other_elements: The list of other montage elements to check for neighbors.
    :type other_elements: List[MontageElement]
    :return: A list of neighboring montage elements.
    :rtype: List[MontageElement]
    """
    neighbors = []
    potential_neighbors = element.build_neighbors()

    for montage_element in other_elements:
        if (montage_element.image_file.x_position, montage_element.image_file.y_position) in potential_neighbors:
            neighbors.append(montage_element)

    return neighbors
