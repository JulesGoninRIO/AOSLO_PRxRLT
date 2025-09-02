from typing import List, Tuple
from pathlib import Path
import cv2
import numpy as np

from src.plotter.plotter import Plotter
from src.cell.analysis.constants import PIXELS_PER_DEGREE
from src.cell.cell_detection.constants import IMAGE_CENTER, IMAGE_SIZE, DISTANCE_FROM_EDGE
from src.cell.montage.montage_element import RegionOfInterest
from src.cell.montage.montage_mosaic import MontageMosaic

class ConeGathererPlotter(Plotter):
    def __init__(self, path: Path):
        """
        Initialize the ConeGathererPlotter.

        :param path: The path where the output will be saved.
        :type path: Path
        """
        super().__init__(path)
        self.overlap_save_path = self.output_path / "overlapping_regions"
        self.overlap_save_path.mkdir(parents=True, exist_ok=True)
        self.rois_save_path = self.output_path / "regions_of_interest"
        (self.rois_save_path / "mdrnn").mkdir(parents=True, exist_ok=True)
        (self.rois_save_path / "atms").mkdir(parents=True, exist_ok=True)

    def plot_both_cones_montage_image(self, montage_mosaic: MontageMosaic, solved_atms_cones: List[List[int]], solved_mdrnn_cones: List[List[int]]) -> None:
        """
        Plot a montage image with both ATMS and MDRNN solved cones.

        :param montage_mosaic: The montage mosaic object containing the image data.
        :type montage_mosaic: MontageMosaic
        """
        image = cv2.cvtColor(montage_mosaic._mosaic_cache.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for cone in solved_atms_cones:
            cv2.circle(image, (cone[0], cone[1]), radius=1, color=(0, 0, 255), thickness=0)
        for cone in solved_mdrnn_cones:
            cv2.circle(image, (cone[0], cone[1]), radius=1, color=(255, 0, 0), thickness=0)
        cv2.imwrite(str(self.overlap_save_path / f'mosaic.tif'), (image).astype(np.uint8))

    def plot_failed_sampling_image(self, img_name: str, img_array: np.ndarray, filtered_cones: List[np.ndarray], barycenter: np.ndarray | None, name: str) -> None:
        """
        Plot a failed sampling image with detected cones, barycenter if given, center of the image, and a rectangle around the image.

        :param img_name: The name of the image file.
        :type img_name: str
        :param img_array: The image array.
        :type img_array: np.ndarray
        :param filtered_cones: List of detected cone coordinates.
        :type filtered_cones: List[np.ndarray]
        :param barycenter: The barycenter of the detected cones.
        :type barycenter: np.ndarray | None
        :param name: The name of the detection method (e.g., 'mdrnn' or 'atms').
        :type name: str
        """
        for cone in filtered_cones:
            cv2.circle(img_array, tuple(cone), radius=1, color=[255, 0, 0], thickness=0)
        if barycenter is not None:
            cv2.circle(img_array, tuple(np.round(barycenter).astype(int)), radius=1, color=[0, 200, 255], thickness=0)
        cv2.rectangle(img_array, (DISTANCE_FROM_EDGE, DISTANCE_FROM_EDGE), (IMAGE_SIZE - DISTANCE_FROM_EDGE, IMAGE_SIZE - DISTANCE_FROM_EDGE), color=[0, 0, 255], thickness=1)
        cv2.circle(img_array, IMAGE_CENTER, radius=1, color=[0, 255, 0], thickness=0)
        cv2.imwrite(self.rois_save_path / name / img_name, img_array)
    
    def plot_sampling_image(self, img_name: str, img_array: np.ndarray, filtered_cones: List[np.ndarray], rois: List[RegionOfInterest], name: str, dark_region: np.ndarray) -> None:
        """
        Plot a sampling image with detected cones, regions of interest, center of the image, and dark regions.

        :param img_name: The name of the image file.
        :type img_name: str
        :param img_array: The image array.
        :type img_array: np.ndarray
        :param filtered_cones: List of detected cone coordinates.
        :type filtered_cones: List[np.ndarray]
        :param rois: List of regions of interest.
        :type rois: List[RegionOfInterest]
        :param name: The name of the detection method (e.g., 'mdrnn' or 'atms').
        :type name: str
        :param dark_region: The dark region mask.
        :type dark_region: np.ndarray
        """
        img_array[dark_region == 0] = [0, 0, 0]
        for cone in filtered_cones:
            cv2.circle(img_array, tuple(cone), radius=1, color=[255, 0, 0], thickness=0)
        for roi in rois:
            roi.draw(img_array, color=[0, 0, 255])
        cv2.circle(img_array, IMAGE_CENTER, radius=1, color=[0, 255, 0], thickness=0)
        cv2.imwrite(self.rois_save_path / name / img_name, img_array)


    def plot_single_image(
        self,
        img_name: str,
        img_array: np.ndarray,
        cones: List[List[int]],
        cones_solved: List[List[int]],
        overlap_regions: List[Tuple[int,int,int,int]]) -> None:
        """
        Plot a single image with cones and overlap regions.

        :param image_key: The key of the image file.
        :type image_key: str
        :param cones: List of cone coordinates.
        :type cones: List[List[int]]
        :param cones_solved: List of solved cone coordinates.
        :type cones_solved: List[List[int]]
        :param overlap_regions: Tuple containing the coordinates of overlap regions.
        :type overlap_regions: Tuple[int,int,int,int]
        """
        for (x1, y1, x2, y2) in overlap_regions:
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for cone in cones_solved:
            cv2.circle(img_array, (cone[0], cone[1]), radius=1, color=[255, 0, 0], thickness=0)
        for cone in cones:
            cv2.circle(img_array, (cone[1], cone[0]), radius=0, color=[0, 0, 255], thickness=0)
        cv2.imwrite(str(self.overlap_save_path / (img_name + '_cones.tif')), img_array)

    def plot_montage_image(
        self,
        montage_mosaic: MontageMosaic,
        solved_cones: List[List[int]],
        cones: List[List[int]],
        name: str):
        """
        Plot a montage image with solved cones and cones.

        :param montage_mosaic: The montage mosaic object containing the image data.
        :type montage_mosaic: MontageMosaic
        :param solved_cones: List of solved cone coordinates.
        :type solved_cones: List[Tuple[int]]
        :param cones: List of cone coordinates.
        :type cones: List[Tuple[int]]
        :param name: The name of the montage (e.g., 'atms' or 'mdrnn').
        :type name: str
        """
        image = cv2.cvtColor(montage_mosaic._mosaic_cache.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for cone in solved_cones:
            cv2.circle(image, (cone[0], cone[1]), radius=1, color=(0, 0, 255), thickness=0)
        for cone in cones:
            cv2.circle(image, (round(cone[0]), round(cone[1])), radius=0, color=(255, 0, 0), thickness=0)
        if name == 'atms':
            self.atms_solved_cones = solved_cones
        elif name == 'mdrnn':
            self.mdrnn_solved_cones = solved_cones
        cv2.imwrite(str(self.overlap_save_path / f'{name}_mosaic.tif'), (image).astype(np.uint8))
