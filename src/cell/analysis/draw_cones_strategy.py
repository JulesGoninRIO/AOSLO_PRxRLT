from abc import ABC, abstractmethod
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads

from src.cell.cell_detection.cone_gatherer import ConeGatherer
from src.cell.montage.montage_mosaic import MontageMosaic

class DrawConesStrategy(ABC):
    """
    Abstract base class for drawing cones strategy.

    Methods
    -------
    draw():
        Abstract method to draw cones.
    _get_image_names():
        Method to get image names.
    """
    def __init__(self):
        """
        Initializes the DrawConesStrategy class.
        """
        pass

    @abstractmethod
    def draw(self):
        """
        Abstract method to draw cones.

        This method should be implemented by subclasses to define how cones are drawn.
        """
        pass

    def _get_image_names(self):
        """
        Get image names.

        This method should be implemented to return the names of images used in the strategy.
        """
        pass

def is_point_in_polygons(point_to_check, polygon):
    """
    Check if a point is within a polygon.

    This function checks if the given point is within the specified polygon.

    :param point_to_check: The point to check.
    :type point_to_check: tuple or list
    :param polygon: The polygon to check against.
    :type polygon: shapely.geometry.Polygon or str
    :return: True if the point is within the polygon, False otherwise.
    :rtype: bool
    """
    if polygon != 'POLYGON EMPTY':
        if polygon.contains(Point(point_to_check)):
            return True
    return False

class DrawConesOnConfocal(DrawConesStrategy):
    """
    A class to draw cones on confocal images.

    Methods
    -------
    draw(image_mosaic):
        Draws cones on the provided image mosaic.
    """
    def draw(self, image_mosaic: MontageMosaic):
        """
        Draw cones on the provided image mosaic.

        This method sets the overlap regions for the image mosaic, gathers cone data from ATM files,
        and processes each image to draw cones on the confocal images.

        :param image_mosaic: The image mosaic to draw cones on.
        :type image_mosaic: MontageMosaic
        :return: None
        """
        print("Drawing cones on confocal")
        image_mosaic.set_overlap_regions()
        # image_mosaic.draw_overlap_regions_on_images()
        dir = Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject100\Session466\postprocessed_atms_single')
        cone_gatherer = ConeGatherer(dir)
        cones_per_image = cone_gatherer.get_cones_from_atms()
        for csv_image_file, cones in cones_per_image.items():
            image_file = csv_image_file.to_image_file()
            # image_file.read_data(dir.parent)
            montage_element = image_mosaic.find_montage_element_by_image_file(image_file)
            dark_regions =montage_element.image_file.dark_regions
            cone_centers = np.argwhere(cones)
            new_centers = []
            for cone_center in cone_centers:
                cone_center.transform(montage_element.affine_transform)
                for neighbor_name, polygon in image_mosaic.overlap_regions[str(image_file)].items():
                    if is_point_in_polygons(cone_center, image_mosaic.overlap_regions[str(image_file)]):
                        if dark_regions[cone_center[0], cone_center[1]]:
                            new_centers.append(cone_center)
                    else:
                        if dark_regions[cone_center[0], cone_center[1]]:
                            new_centers.append(cone_center)
                # if not (cone_center == image_mosaic.overlap_regions[str(image_file)]).all(axis=1).any():
                #     if binary_map[cone_center[0], cone_center[1]]:
                #         new_centers.append(cone_center)
        #     draw_cones()
        #     centers = np.array(new_centers)
        #     for center in centers:
        #         image_cones = cv2.circle(image, (center[1], center[0]),
        #                                     radius=0, color=(255, 0, 0), thickness=3)
        #     if voronoi:
        #         try:
        #             VoronoiDiagram(centers, image).draw_voronoi(image_name,
        #                                                         binary_map, overlap_region,
        #                                                         area_to_analyze_image,
        #                                                         self.__draw_path)
        #         except RuntimeError:
        #             pass
        #         out_name = os.path.splitext(
        #             image_name)[0] + "_voronoi_cones.tif"
        #     else:
        #         out_name = os.path.splitext(
        #             image_name)[0] + "_cones.tif"
        # cv2.imwrite(os.path.join(
        #     self.__draw_path, out_name), image_cones)

class DrawConesCalculatedSplit(DrawConesStrategy):
    def draw(self):
        print("Drawing cones on calculated split")

class DrawConesOnConfocalWithVoronoi(DrawConesStrategy):
    def draw(self):
        print("Drawing cones on confocal with Voronoi")

class DrawConesCalculatedSplitWithVoronoi(DrawConesStrategy):
    def draw(self):
        print("Drawing cones on calculated split with Voronoi")



