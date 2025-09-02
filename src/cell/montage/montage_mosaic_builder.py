from pathlib import Path
import os
import re
import tifffile
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
import cv2
import imageio.v2 as imageio
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import logging
from pathlib import Path
import time
from multiprocessing import Manager, Process, Lock
import math

from src.configs.parser import Parser
from src.shared.datafile.datafile_constants import ImageModalities
from src.shared.datafile.image_file import ImageFile
from src.shared.computer_vision.point import Point
from src.shared.computer_vision.square import Square
from src.shared.computer_vision.image import get_boundaries
from src.cell.affine_transform import AffineTransform
from src.cell.processing_path_manager import ProcessingPathManager
from src.cell.montage.montage_mosaic import MontageMosaic, get_corners, correct_transforms
from src.cell.montage.montage_element import MontageElement
from src.cell.montage.constants import IMAGE_SIZE
from src.cell.montage.matlab_reader import MatlabReader
from src.cell.montage.ssim_processing import SSIMProcessing

class MontageMosaicBuilder(ABC):
    """
    Abstract base class for building a montage mosaic.

    :param path_manager: The path manager for handling file paths.
    :type path_manager: Path
    """
    def __init__(self, path_manager: ProcessingPathManager):
        """
        Initialize the MontageMosaicBuilder.

        :param path_manager: The path manager for handling file paths.
        :type path_manager: Path
        """
        self.elements = []
        self.path_manager = path_manager

    def add_element(self, image_file: ImageFile, fundus_transform: AffineTransform | None = None, transform: AffineTransform | None = None):
        """
        Add an element to the montage mosaic.

        :param image_file: The image file to add.
        :type image_file: ImageFile
        :param fundus_transform: The fundus transform to apply to the image file.
        :type fundus_transform: AffineTransform, optional
        :param transform: The affine transform to apply to the image file.
        :type transform: AffineTransform, optional
        """
        element = MontageElement(image_file, self.path_manager, fundus_transform=fundus_transform, transform=transform)
        self.elements.append(element)

    def build(
        self,
        has_been_corrected: bool,
        center_point: Point = None,
        mosaic_shape: Tuple[int, int] = None,
        ratio: Tuple[float, float] = None,
        fundus_shape: Tuple[int, int] = None
    ) -> MontageMosaic:
        """
        Build the montage mosaic.

        :param has_been_corrected: Flag indicating if the mosaic has been corrected.
        :type has_been_corrected: bool
        :param center_point: The center point of the mosaic.
        :type center_point: Point, optional
        :param mosaic_shape: The shape of the mosaic.
        :type mosaic_shape: Tuple[int, int], optional
        :param ratio: The ratio to apply to the transform.
        :type ratio: Tuple[float, float], optional
        :param fundus_shape: The shape of the fundus.
        :type fundus_shape: Tuple[int, int], optional
        :return: The built montage mosaic.
        :rtype: MontageMosaic
        """
        self.mosaic = MontageMosaic(self.path_manager, has_been_corrected, center_point, mosaic_shape, ratio, fundus_shape)
        for element in self.elements:
            self.mosaic.add_element(element)
        corners = get_corners(self.mosaic.montage_elements, ratio, self.path_manager.path)
        correct_transforms(self.mosaic.montage_elements, corners)
        self.mosaic.transformed_center = Point(self.mosaic.transformed_center.x - corners[2], self.mosaic.transformed_center.y - corners[3])
        return self.mosaic

    @abstractmethod
    def build_mosaic(self):
        """
        Abstract method to build the mosaic.

        This method should be implemented by subclasses to define the specific
        logic for building the mosaic.
        """
        pass

class MatlabMontageMosaicBuilder(MontageMosaicBuilder):
    """
    Builder class for creating a montage mosaic using Matlab data.

    This class reads transformation data from Matlab, applies the necessary transformations,
    and builds the montage mosaic.

    :param path_manager: The path manager for handling file paths.
    :type path_manager: Path
    """
    # TODO: If we get directly from matlab, try to guess the center with fundus

    def build_mosaic(self) -> MontageMosaic:
        """
        Build the montage mosaic using Matlab data.

        This method reads the transformation data from Matlab, applies the necessary transformations,
        and constructs the montage mosaic.

        :return: The built montage mosaic.
        :rtype: MontageMosaic
        """
        self.matlab = MatlabReader(self.path_manager.montage.path)
        # don't ask why all those transformations, they come from Matlab's montage code
        transforms = self.matlab.get_transforms()
        mosaic_shape = (round(self.matlab.data['maxYAll'][0][0]), round(self.matlab.data['maxXAll'][0][0]))
        global_transform = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-np.float64(self.matlab.data['minXAll'][0][0]), -np.float64(self.matlab.data['minYAll'][0][0]), 1]
        ])
        for i, transform in enumerate(transforms):
            val_name = self.matlab.names[i][0]
            H = np.linalg.pinv(transform.T)
            H[:, 2] = [0, 0, 1]
            tform_matrix = np.dot(H, global_transform)
            image_file = ImageFile(val_name)
            image_file.read_data(self.path_manager.path)
            self.add_element(image_file, AffineTransform.from_cv2(tform_matrix.T[:2]))
        return self.build(False, mosaic_shape=mosaic_shape)

class CorrectedMontageMosaicBuilder(MontageMosaicBuilder):
    """
    Builder class for creating a corrected montage mosaic.

    This class reads corrected transformation data, applies the necessary transformations,
    and builds the corrected montage mosaic.

    :param path_manager: The path manager for handling file paths.
    :type path_manager: ProcessingPathManager
    """
    def __init__(self, path_manager: ProcessingPathManager):
        """
        Initialize the CorrectedMontageMosaicBuilder.

        :param path_manager: The path manager for handling file paths.
        :type path_manager: ProcessingPathManager
        """
        super().__init__(path_manager)
        self.corrected_montage_filename = Parser.get_corrected_montage_filename()
        self.path_manager = self.path_manager
        self.center_point = None
        self.component_locations = None
        self.size = None
        self.fundus_shape = None
        self.ratio = None
        self.correct_ssim = True # Parser.correct_ssim()

    def build_mosaic(self) -> MontageMosaic:
        """
        Build the corrected montage mosaic.

        This method reads the corrected transformation data, applies the necessary transformations,
        and constructs the corrected montage mosaic.

        :return: The built corrected montage mosaic.
        :rtype: MontageMosaic
        """
        self.read_corrected_file()
        self.matlab = MatlabReader(self.path_manager.montage.path)
        self.big_image = np.zeros((self.fundus_shape[0], self.fundus_shape[1]), dtype=np.uint8)
        for key, values in self.matlab.matched_chains.items():
            ref_key = str(int(key) + 1)
            # if ref_key == '37':
            #     print("oe")
            ref_name = f'ref_{ref_key}_combined_m1.tif'
            try:
                row = self.component_locations[self.component_locations['name'] == ref_name].iloc[-1]
            except IndexError:
                logging.error(f"Error retrieving row with {ref_name}")
                continue
            try:
                if np.isnan(eval(row['x'])) or np.isnan(row['y']):
                    continue
                self.location_confocal = (float(row['x']) - self.size[0] / 2, float(row['y']) - self.size[1] / 2)
            except Exception as e:
                logging.error(f"Error {repr(e)} with {ref_name}")
                continue
            angle = row['angle']
            for val in values:
                val_name = self.matlab.names[val][0]
                # print(val_name)
                transform = self.get_element_transform(val_name, ref_key, angle)
                if tuple(transform.transform_points(np.array([0,0])).squeeze()) == (0,0):
                    print(ref_name, val_name, self.location_confocal)
                self.add_element(ImageFile(val_name), fundus_transform=transform)
                print(ref_name, val_name, self.location_confocal, tuple(transform.transform_points(np.array([0,0])).squeeze()))
        # from src.cell.montage.montage_element import get_corners
        # corners = get_corners(self.elements, self.ratio, self.base_path)
        # if self.correct_ssim:
        #     # ratio:
        #     from src.cell.montage.montage_element import get_corners
        #     corners = get_corners(self.elements, self.ratio, self.base_path)
        #     SSIMProcessing(self.elements, self.center_point, self.base_path, self.matlab, self.ratio, corners).improve_montage_correction_with_ssim()
        # cv2.imwrite(r'C:\Users\BardetJ\Downloads\tests4.tif', self.big_image)
        return self.build(True, center_point=self.center_point, ratio=self.ratio, fundus_shape=self.fundus_shape) #, mosaic_shape = (corners[0], corners[1]))

    def read_corrected_file(self) -> None:
        """
        Read the corrected file from the GUI results.

        This method reads the corrected transformation data from a CSV file generated by the GUI,
        extracts the center point, component locations, image size, and fundus shape.

        :raises FileNotFoundError: If the corrected file is not found.
        """
        logging.info("Read the GUI results")
        try:
            locations = pd.read_csv(
                str(self.path_manager.montage.path / self.corrected_montage_filename),
                header=None,
                names=['name', 'x', 'y', 'image_size', 'i', 'angle']
            )
        except FileNotFoundError:
            logging.error(f"Cannot find the location file from the GUI step. \
                Please place the output file from the GUI ({self.corrected_montage_filename}) \
                in the {str(self.path_manager.montage.path)} or do the GUI for montaging if not done.")
            raise
        center_location = locations.iloc[0]
        center_point = eval(center_location['name'])
        self.center_point = Point.from_cartesian(x=center_point[0], y=center_point[1])
        self.component_locations = locations.iloc[1:]
        angle_argmin = self.component_locations['angle'].abs().argmin()
        self.size = eval(self.component_locations['image_size'].iloc[angle_argmin])
        if (angle := self.component_locations['angle'].iloc[angle_argmin]) != 0:
            # if angle is non-zero, rotate self.size to get the real image size
            self.size = (
                1 + math.ceil(self.size[0] * math.cos(math.radians(angle)) \
                            - self.size[1] * math.sin(math.radians(angle))),
                1 + math.ceil(self.size[1] * math.cos(math.radians(angle)) \
                            - self.size[0] * math.sin(math.radians(angle)))
            )
        self.fundus_shape = (eval(center_location['x'])[0], eval(center_location['x'])[1])

    def get_element_transform(self, name: str, ref_key: str, angle: int) -> AffineTransform:
        """
        Get the transformation matrix for an image element.

        This method loads and prepares the image, computes the combined transformation matrix,
        applies the transformation, pads the image to the fundus shape, and computes the perspective
        transformation matrix.

        :param name: The name of the image file.
        :type name: str
        :param ref_key: The reference key for alignment.
        :type ref_key: str
        :param angle: The rotation angle.
        :type angle: int
        :return: The affine transformation.
        :rtype: AffineTransform
        """
        if angle!=0:
            print(angle)
        image = self.load_and_prepare_image(name, ref_key)
        combined_matrix = self.get_combined_transformation_matrix(image.shape, angle)
        transformed_image = cv2.warpPerspective(image, combined_matrix, (int(max(0, combined_matrix[0, 2])+image.shape[1]), int(max(0, combined_matrix[1, 2])+image.shape[0])))
        padded_image = self.pad_image_to_fundus_shape(transformed_image)
        try:
            self.big_image += padded_image[:,:,0]
        except ValueError:
            self.big_image = np.pad(self.big_image, ((0, max(0,padded_image.shape[0] - self.big_image.shape[0])), (0, max(0, padded_image.shape[1] - self.big_image.shape[1]))), mode='constant')
            padded_image_copy = padded_image.copy()
            padded_image_copy = np.pad(padded_image_copy, ((0, max(0,self.big_image.shape[0] - padded_image.shape[0])), (0, max(0, self.big_image.shape[1] - padded_image.shape[1])), (0,0)), mode='constant')
            self.big_image += padded_image_copy[:,:,0]
        box_sorted = self.get_sorted_bounding_box(padded_image)
        matrix = self.get_perspective_transform_matrix(box_sorted)
        transform = AffineTransform.from_cv2(matrix[:2])
        return transform

    def load_and_prepare_image(self, name: str, ref_key: str) -> np.ndarray:
        """
        Load and prepare the image for transformation.

        This method loads the image and resizes it to the required dimensions.

        :param name: The name of the image file.
        :type name: str
        :param ref_key: The reference key for alignment.
        :type ref_key: str
        :return: The loaded and resized image.
        :rtype: np.ndarray
        """
        image = self.load_image(name, ref_key)
        # self.shape_for_rotation = (image.shape[1], image.shape[2])
        return self.resize_image(np.transpose(image, (1, 2, 0)))

    def get_combined_transformation_matrix(self, image_shape: Tuple[int, int], angle: int) -> np.ndarray:
        """
        Get the combined transformation matrix.

        This method computes the scaling, rotation, and translation matrices and combines them
        into a single transformation matrix.

        :param image_shape: The shape of the image.
        :type image_shape: Tuple[int, int]
        :param angle: The rotation angle.
        :type angle: int
        :return: The combined transformation matrix.
        :rtype: np.ndarray
        """
        scaling_matrix = self.get_scaling_matrix(image_shape)
        image_center = np.array([scaling_matrix[0][0] * image_shape[1], scaling_matrix[1, 1] * image_shape[0]]) / 2
        rotation_matrix = self.get_rotation_matrix(image_center, angle)
        # rotation_matrix[:, 2] = [0,0,1]
        # rotation_matrix[0, 2] = -rotation_matrix[0, 2]
        # rotation_matrix[1, 2] = rotation_matrix[1, 2]
        # rotation_matrix, new_position = self.rotate_and_displace(image_shape, self.shape_for_rotation, angle)
        # rotation_matrix[:, 2] = new_position
        translation_matrix = self.get_translation_matrix(scaling_matrix)
        combined_matrix = scaling_matrix @ translation_matrix @ rotation_matrix
        return combined_matrix

    def rotate_and_displace(self, image_shape, padded_shape, angle):
        """
        Calculate the rotation and displacement of the original image within the padded image.

        :param image_shape: The shape of the original image (height, width).
        :param padded_shape: The shape of the padded image (height, width).
        :param angle: The rotation angle in degrees.
        :return: The transformation matrix and the new position of the original image.
        """
        # Calculate the center of the original image
        image_center = (image_shape[1] / 2, image_shape[0] / 2)

        # Calculate the center of the padded image
        padded_center = (padded_shape[1] / 2, padded_shape[0] / 2)

        # Calculate the offset due to padding
        offset = (padded_center[0] - image_center[0], padded_center[1] - image_center[1])

        # Get the rotation matrix for the padded image
        rotation_matrix = self.get_rotation_matrix(padded_center, angle)

        # Apply the rotation matrix to the offset to find the new position
        new_position = np.dot(rotation_matrix[:, :2], np.array([offset[0], offset[1]])) + rotation_matrix[:, 2]

        return rotation_matrix, new_position

    def get_sorted_bounding_box(self, image: np.ndarray) -> Tuple[np.ndarray]:
        """
        Get the sorted bounding box of the non-transparent pixels.

        This method computes the bounding box of the non-transparent pixels and sorts the corners.

        :param image: The input image.
        :type image: np.ndarray
        :return: The sorted bounding box.
        :rtype: Tuple[np.ndarray]
        """
        points = self.get_non_transparent_pixels(image)
        box = self.get_bounding_box(points)
        box_sorted = self.sort_swapped_corners(box)
        return box_sorted

    def get_perspective_transform_matrix(self, box_sorted: Tuple[np.ndarray]) -> np.ndarray:
        """
        Get the perspective transformation matrix.

        This method computes the perspective transformation matrix from the sorted bounding box.

        :param box_sorted: The sorted bounding box.
        :type box_sorted: Tuple[np.ndarray]
        :return: The perspective transformation matrix.
        :rtype: np.ndarray
        """
        top_left, top_right, bottom_left, bottom_right = self.get_corners(box_sorted)
        src_points = np.array([[0, 0], [720, 0], [0, 720], [720, 720]], dtype=np.float32)
        dst_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
        return cv2.getPerspectiveTransform(src_points, dst_points)

    def load_image(self, name: str, ref_key: str) -> np.ndarray:
        """
        Load the image from the file system.

        This method loads the image file based on the given name and reference key.

        :param name: The name of the image file.
        :type name: str
        :param ref_key: The reference key for alignment.
        :type ref_key: str
        :return: The loaded image.
        :rtype: np.ndarray
        """
        image_path = self.path_manager.montage.path / f'{name[:-4]}_aligned_to_ref{ref_key}_m1.tif'
        image = imageio.imread(str(image_path))
        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to the required dimensions.

        This method resizes the image based on the predefined size and ratio.

        :param image: The input image.
        :type image: np.ndarray
        :return: The resized image.
        :rtype: np.ndarray
        """
        if not self.ratio:
            self.ratio = (image.shape[1] / self.size[0], image.shape[0] / self.size[1])
        return cv2.resize(image, (round(self.size[0]), round(self.size[1])))

    def get_scaling_matrix(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get the scaling matrix.

        This method computes the scaling matrix based on the image shape and predefined size.

        :param image_shape: The shape of the image.
        :type image_shape: Tuple[int, int]
        :return: The scaling matrix.
        :rtype: np.ndarray
        """
        scale_x = self.size[0] / image_shape[1]
        scale_y = self.size[1] / image_shape[0]
        scaling_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64)
        return scaling_matrix

    def get_rotation_matrix(self, image_center: np.ndarray, angle: int) -> np.ndarray:
        """
        Get the rotation matrix.

        This method computes the rotation matrix based on the image center and rotation angle.

        :param image_center: The center of the image.
        :type image_center: np.ndarray
        :param angle: The rotation angle.
        :type angle: int
        :return: The rotation matrix.
        :rtype: np.ndarray
        """
        rotation_matrix = cv2.getRotationMatrix2D((image_center[0], image_center[1]), angle, 1)
        rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
        return rotation_matrix

    def get_translation_matrix(self, scaling_matrix: np.ndarray) -> np.ndarray:
        """
        Get the translation matrix.

        This method computes the translation matrix based on the scaling matrix and location.

        :param scaling_matrix: The scaling matrix.
        :type scaling_matrix: np.ndarray
        :return: The translation matrix.
        :rtype: np.ndarray
        """
        translation_matrix = np.array([[1, 0, self.location_confocal[0] / scaling_matrix[0][0]],
                                    [0, 1, self.location_confocal[1] / scaling_matrix[1][1]],
                                    [0, 0, 1]], dtype=np.float64)
        return translation_matrix

    def pad_image_to_fundus_shape(self, image: np.ndarray) -> np.ndarray:
        """
        Pad the image to the fundus shape.

        This method pads the image to match the predefined fundus shape.

        :param image: The input image.
        :type image: np.ndarray
        :return: The padded image.
        :rtype: np.ndarray
        """
        pad_y = max(self.fundus_shape[1] - image.shape[0], 0)
        pad_x = max(self.fundus_shape[0] - image.shape[1], 0)
        padded_image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)))
        return padded_image

    @staticmethod
    def get_non_transparent_pixels(image: np.ndarray) -> np.ndarray:
        """
        Get the non-transparent pixels from the image.

        This method extracts the coordinates of the non-transparent pixels from the image.

        :param image: The input image.
        :type image: np.ndarray
        :return: The coordinates of the non-transparent pixels.
        :rtype: np.ndarray
        """
        alpha = image[:, :, 1]
        points = np.column_stack(np.nonzero(alpha))
        return points

    @staticmethod
    def sort_swapped_corners(box: np.ndarray) -> np.ndarray:
        """
        Sort the swapped corners of the bounding box.

        This method sorts the corners of the bounding box to ensure correct order.

        :param box: The bounding box.
        :type box: np.ndarray
        :return: The sorted bounding box.
        :rtype: np.ndarray
        """
        box_swapped = box[:, ::-1]
        box_sorted = box_swapped[box_swapped[:, 0].argsort()]
        return box_sorted

    @staticmethod
    def get_non_transparent_pixels(image: np.ndarray) -> np.ndarray:
        """
        Get non-transparent pixels from an image.

        This method extracts the non-transparent pixels from the given image based on the alpha channel.

        :param image: The input image with an alpha channel.
        :type image: np.ndarray
        :return: An array of points representing the non-transparent pixels.
        :rtype: np.ndarray
        """
        alpha = image[:, :, 1]
        points = np.column_stack(np.nonzero(alpha))
        return points

    @staticmethod
    def get_bounding_box(points: np.ndarray) -> np.ndarray:
        """
        Get the bounding box for a set of points.

        This method calculates the minimum area bounding box for the given set of points.

        :param points: The input points.
        :type points: np.ndarray
        :return: An array representing the bounding box.
        :rtype: np.ndarray
        """
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        return box

    @staticmethod
    def sort_swapped_corners(box: np.ndarray) -> np.ndarray:
        """
        Sort the corners of a bounding box.

        This method sorts the corners of the bounding box after swapping the coordinates.

        :param box: The input bounding box.
        :type box: np.ndarray
        :return: An array representing the sorted corners of the bounding box.
        :rtype: np.ndarray
        """
        box_swapped = box[:, ::-1]
        box_sorted = box_swapped[box_swapped[:, 0].argsort()]
        return box_sorted

    @staticmethod
    def get_corners(box_sorted: np.ndarray) -> Tuple[np.ndarray]:
        """
        Get the corners of a bounding box.

        This method extracts the top-left, top-right, bottom-left, and bottom-right corners
        from the sorted bounding box.

        :param box_sorted: The sorted bounding box.
        :type box_sorted: np.ndarray
        :return: A tuple containing the top-left, top-right, bottom-left, and bottom-right corners.
        :rtype: Tuple[np.ndarray]
        """
        top_left = box_sorted[0] if box_sorted[0][1] <= box_sorted[1][1] else box_sorted[1]
        bottom_left = box_sorted[0] if box_sorted[0][1] >= box_sorted[1][1] else box_sorted[1]
        top_right = box_sorted[2] if box_sorted[2][1] <= box_sorted[3][1] else box_sorted[3]
        bottom_right = box_sorted[2] if box_sorted[2][1] >= box_sorted[3][1] else box_sorted[3]
        return top_left, top_right, bottom_left, bottom_right

class RawImageMosaicBuilder(MontageMosaicBuilder):
    """
    Builder class for creating a raw image mosaic.

    This class reads transformation data directly from a montage image, applies the necessary transformations,
    and builds the raw image mosaic.

    :param path_manager: The path manager for handling file paths.
    :type path_manager: Path
    """
    # get from a montage image directly, if .mat file is missing
    # TODO: correct

    def build_mosaic(self) -> MontageMosaic:
        """
        Build the raw image mosaic.

        This method reads the transformation data directly from a montage image, applies the necessary transformations,
        and constructs the raw image mosaic.

        :return: The built raw image mosaic.
        :rtype: MontageMosaic
        """
        # don't ask why all those transformations, they come from Matlab's montage code
        transforms = self.matlab.get_transforms()
        height = self.matlab.data['maxYAll'][0][0]
        width = self.matlab.data['maxXAll'][0][0]
        global_transform = np.array([[1, 0, 0], [0, 1, 0], [-np.float64(self.matlab.data['minXAll'][0][0]), -np.float64(self.matlab.data['minYAll'][0][0]), 1]])
        for i, transform in enumerate(transforms):
            val_name = self.matlab.names[i][0]
            H = np.linalg.pinv(transform.T)
            H[:, 2] = [0, 0, 1]
            tform_matrix = np.dot(H, global_transform)
            image_file = ImageFile(val_name)
            image_file.read_data(self.path_manager.path)
            self.add_element(image_file, AffineTransform.from_cv2(tform_matrix.T[:2]))
        return self.build(False)

    def create_own_shifts(self, modality: ImageModalities = ImageModalities.CO) \
            -> Dict[str, np.array]:
        """
        Returns the shifts of each image from a montage with the form
        [[1,0,dy][0,1,dx]] because MATLAB montage's shifts are wrong and points as (y,x)

        :param input_dir: directory where the montage images are
        :type input_dir: str
        :param modality: the modality to find the shift from, defaults to ImageModalities.CO
        :type modality: ImageModalities, optional
        :return: a dictionnary with the image names as key and the 2x3 matrix as
                the shift of the image
        :rtype: Dict[str, np.array]
        """

        # Let's gather the single images from a specific modality
        filenames = {entry.name for entry in os.scandir(str(self.path_manager.montage.path)) if modality.value in entry.name}
        # Avoid repetitive elements
        filenames = set(filenames)

        # Compile regular expressions for speed
        regex1 = re.compile(r"_aligned_to_ref(\d+)_m(\d+)")
        regex2 = re.compile("CalculatedSplit")

        # We will need to find each shifts from the single images that have all the
        # same size
        dic_list = []
        for image_name in filenames:
            image = tifffile.imread(str(self.path_manager.montage.path / image_name))
            try:
                points = find_points(image)
            except AssertionError:
                continue
            out_name = regex2.sub("Confocal", regex1.sub("", image_name))
            dic_data = {"name": out_name,
                        "top_left_y": points[0][0],
                        "top_left_x": points[0][1],
                        "top_right_y": points[1][0],
                        "top_right_x": points[1][1],
                        "bottom_left_y": points[2][0],
                        "bottom_left_x": points[2][1],
                        "bottom_right_y": points[3][0],
                        "bottom_right_x": points[3][1]}
            dic_list.append(dic_data)

        # If you want ot get the shifts from the entire connected compoenent as well
        # we are gathering their shifts only from translation because the whole
        # component has no rotation
        shifts = pd.DataFrame.from_dict(dic_list)

        return shifts


def find_points(image: np.ndarray, angle_c: int = 0) -> List[List[int]]:
    """
    Find the corners of an image and returns them as [top left, top right,
    bottom left, bottom right]. We have to make sure the image does contain only
    ones or has an alpha chanel, otherwise we can have zeros pixels that belong
    to the image but is not recongized as such.

    :param image: the image to find corners
    :type image: np.ndarray
    :raises AssertionError: the image has no alpha channel or is not composed of
                            only zeros and ones
    :return: the corners of the image
    :rtype: List[List[int]]
    """

    def get_corner_points(filled_pixels, top, bottom, left, right):
        top_left = np.min(filled_pixels[filled_pixels[:, 0] == top][:, 1])
        top_right = np.max(filled_pixels[filled_pixels[:, 0] == top][:, 1])
        bottom_left = np.min(filled_pixels[filled_pixels[:, 0] == bottom][:, 1])
        bottom_right = np.max(filled_pixels[filled_pixels[:, 0] == bottom][:, 1])
        return top_left, top_right, bottom_left, bottom_right

    def rotate_points(angle_c, first_point, second_point, third_point, fourth_point):
        rotations = {
            range(45, 91): (third_point, fourth_point, first_point, second_point),
            range(91, 181): (second_point, third_point, fourth_point, first_point),
            range(-90, -44): (second_point, third_point, fourth_point, first_point),
            range(-180, -89): (third_point, fourth_point, first_point, second_point),
        }
        for angle_range, points in rotations.items():
            if angle_c in angle_range:
                return points
        return first_point, second_point, third_point, fourth_point

    assert len(np.unique(image) == 2), "the image has no alpha channel or is not composed of only 2 values"

    if len(image.shape) > 2:
        image = image[:,:,-1]

    filled_pixels = np.argwhere(image > 0)
    top, bottom, left, right = get_boundaries(image)

    top_left, top_right, bottom_left, bottom_right = get_corner_points(filled_pixels, top, bottom, left, right)

    if abs(top_left - left) < abs(top_right - right):
        first_point = [top, top_left]
        second_point = [np.min(filled_pixels[filled_pixels[:, 1] == right][:, 0]), right]
        third_point = [np.max(filled_pixels[filled_pixels[:, 1] == left][:, 0]), left]
        fourth_point = [bottom, bottom_right]
    else:
        first_point = [np.min(filled_pixels[filled_pixels[:, 1] == left][:, 0]), left]
        second_point = [top, top_right]
        third_point = [bottom, bottom_left]
        fourth_point = [np.max(filled_pixels[filled_pixels[:, 1] == right][:, 0]), right]

    first_point, second_point, third_point, fourth_point = rotate_points(angle_c, first_point, second_point, third_point, fourth_point)

    return [first_point, second_point, third_point, fourth_point]

