from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.plotter.plotter import Plotter
from src.cell.analysis.constants import PIXELS_PER_DEGREE
from src.cell.montage.montage_mosaic import MontageMosaic

class ConeMosaicPlotter(Plotter):
    def __init__(self, montage_mosaic: MontageMosaic, output_path: Path):
        """
        Initialize the ConeMosaicPlotter.

        :param montage_mosaic: The montage mosaic object containing the image data.
        :type montage_mosaic: MontageMosaic
        :param output_path: The path where the output will be saved.
        :type output_path: Path
        """
        super().__init__(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.montage_mosaic = montage_mosaic

    def plot_all_cones(self, cone_locations_mdrnn: List[List[int]], cone_locations_atms: List[List[int]]):
        """
        Plot all cones on the montage image.

        :param cone_locations_mdrnn: List of cone locations detected by MDRNN.
        :type cone_locations_mdrnn: List[List[int]]
        :param cone_locations_atms: List of cone locations detected by ATMS.
        :type cone_locations_atms: List[List[int]]
        """
        img_array = cv2.cvtColor((self.montage_mosaic._mosaic_cache).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_array = self.draw_cones_on_image(img_array, cone_locations_atms, [0, 0, 255])
        img_array = self.draw_cones_on_image(img_array, cone_locations_mdrnn, [255, 0, 0])
        cv2.imwrite(str(self.output_path / 'all_cones.tif'), img_array.astype(np.uint8))

    def plot_cones(self, cone_locations: List[List[int]], color: List[int] = [255, 0, 0], name: str = ''):
        """
        Plot cones on the montage image with a specified color.

        :param cone_locations: List of cone locations.
        :type cone_locations: List[List[int]]
        :param color: Color of the cones to be plotted.
        :type color: List[int]
        :param name: Name for the output file.
        :type name: str
        """
        img_array = cv2.cvtColor((self.montage_mosaic._mosaic_cache).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_array = self.draw_cones_on_image(img_array, cone_locations, color)
        cv2.imwrite(str(self.output_path / f'{name}_cones.tif'), img_array.astype(np.uint8))

    def plot_cones_with_rois(self, solved_mdrnn_cones: List[List[int]], solved_atms_cones: List[List[int]], path: Path | None = None) -> None:
        img_array = cv2.cvtColor(self.montage_mosaic._mosaic_cache.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # img_array[montage_mosaic.binary_map() == 0] = [0, 0, 0]
        for radius in [3/PIXELS_PER_DEGREE, 2, 4, 6, 8, 10, 12]:
            cv2.circle(img_array, tuple(self.montage_mosaic.transformed_center.round()), radius=int(radius*PIXELS_PER_DEGREE), color=[63, 255, 63], thickness=1)
        for cone in solved_atms_cones:
            cv2.circle(img_array, (cone[0], cone[1]), radius=1, color=(0, 0, 255), thickness=0)
        for cone in solved_mdrnn_cones:
            cv2.circle(img_array, (cone[0], cone[1]), radius=1, color=(255, 0, 0), thickness=0)
        for montage_element in self.montage_mosaic.montage_elements:
            for roi in montage_element.roi_atms:
                roi.draw(img_array, color=[0, 127, 255])
            for roi in montage_element.roi_mdrnn:
                roi.draw(img_array, color=[255, 127, 0])
        if path is None:
            path = self.output_path
        cv2.imwrite(str(path / 'cones_with_sampled_rois.tif'), img_array.astype(np.uint8))

    def draw_cones_on_image(self, img_array: np.ndarray, cone_locations: List[List[int]], color: List[int]) -> np.ndarray:
        """
        Draw cones on the given image array.

        :param img_array: The image array on which cones will be drawn.
        :type img_array: np.ndarray
        :param cone_locations: List of cone locations.
        :type cone_locations: List[Tuple[int]]
        :param color: Color of the cones to be drawn.
        :type color: List[int]
        :return: The image array with cones drawn on it.
        :rtype: np.ndarray
        """
        for cone in cone_locations:
            img_array = cv2.circle(
                img_array,
                (round(cone[0]), round(cone[1])),
                radius=1,
                color=color,
                thickness=-1
            )
        return img_array

    def plot_with_distance(self, cones_location_with_distance: Dict[float, np.ndarray], name: str = ''):
        """
        Plot cones on the montage image with distance-based coloring.

        :param cones_location_with_distance: Dictionary of cone locations with distances as keys.
        :type cones_location_with_distance: Dict[float, np.ndarray]
        :param name: Name for the output file.
        :type name: str
        """
        colors = plt.get_cmap('hsv', len(cones_location_with_distance))
        sorted_items = sorted(cones_location_with_distance.items(), key=lambda item: item[0])
        img_array = cv2.cvtColor((self.montage_mosaic._mosaic_cache).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, (_, locations) in enumerate(sorted_items):
            color = colors(i)[:3]
            for x, y in locations:
                try:
                    cv2.circle(img_array, (x, y), 3, np.array(color) * 255, -1)
                except IndexError:
                    continue
        img_with_indication_map = add_indication_map(img_array, cones_location_with_distance)
        cv2.imwrite(str(self.output_path / f'{name}_cones.tif'), img_with_indication_map.astype(np.uint8))

def add_indication_map(img_array: np.ndarray, cones_location: Dict[float, np.ndarray]) -> np.ndarray:
    """
    Add an indication map to the image showing distances.

    :param img_array: The image array to which the indication map will be added.
    :type img_array: np.ndarray
    :param cones_location: Dictionary of cone locations with distances as keys.
    :type cones_location: Dict[float, np.ndarray]
    :return: The image array with the indication map added.
    :rtype: np.ndarray
    """
    margin = 10
    rect_height = 20
    rect_width = 100
    space_between_rects = 2
    total_height = len(cones_location) * (rect_height + space_between_rects)

    start_y = margin
    start_x = img_array.shape[1] - rect_width - margin
    img_with_indication = img_array.copy()
    colors = plt.get_cmap('hsv', len(cones_location))

    # Sort items by distance
    sorted_items = sorted(cones_location.items(), key=lambda item: item[0])

    # Draw rectangles and text for each distance
    for i, (distance, locations) in enumerate(sorted_items):
        color = np.array(colors(i)[:3]) * 255
        # Draw rectangle
        cv2.rectangle(
            img_with_indication, (start_x, start_y + i * (rect_height + space_between_rects)),
            (start_x + rect_width, start_y + rect_height + i * (rect_height + space_between_rects)),
            color.astype(int).tolist(), -1
        )
        # Calculate text size
        text_size = cv2.getTextSize(str(distance), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = start_x + (rect_width - text_size[0]) // 2
        text_y = start_y + i * (rect_height + space_between_rects) + (rect_height + text_size[1]) // 2
        # Put distance text inside the rectangle
        cv2.putText(img_with_indication, str(distance), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return img_with_indication
