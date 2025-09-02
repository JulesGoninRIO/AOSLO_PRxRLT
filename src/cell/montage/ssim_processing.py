from pathlib import Path
import os
import re
import tifffile
import numpy as np
from typing import List, Tuple, Dict, Union
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
from typing import List, Tuple

from src.configs.parser import Parser
from src.shared.datafile.datafile_constants import ImageModalities
from src.shared.datafile.image_file import ImageFile
from src.shared.computer_vision.point import Point
from src.shared.computer_vision.square import Square
from src.shared.computer_vision.image import get_boundaries
from src.cell.montage.montage_mosaic import MontageMosaic
from src.cell.montage.montage_element import MontageElement
from src.cell.affine_transform import AffineTransform
from src.cell.montage.matlab_reader import MatlabReader
from src.cell.montage.ssim import SSIM
from src.cell.montage.montage_element import get_element_neighbors

IMAGE_SIZE = 720

class SSIMProcessing:
    """
    Handles the SSIM processing for montage correction.

    This class processes montage elements to improve their alignment using SSIM (Structural Similarity Index).

    :param elements: List of montage elements to be processed.
    :type elements: List[MontageElement]
    :param center_point: The center point of the montage.
    :type center_point: Point
    :param base_path: The base path for saving results.
    :type base_path: Path
    :param matlab: The Matlab reader object containing matched chains and names.
    :type matlab: MatlabReader
    :param ratio: The ratio for scaling transformations.
    :param corners: The corners of the montage.
    """
    def __init__(self, elements: List[MontageElement], center_point: Point, base_path: Path, matlab: MatlabReader, ratio, corners):
        """
        Initialize the SSIMProcessing class.

        This method initializes the SSIM processing with the given parameters.

        :param elements: List of montage elements to be processed.
        :type elements: List[MontageElement]
        :param center_point: The center point of the montage.
        :type center_point: Point
        :param base_path: The base path for saving results.
        :type base_path: Path
        :param matlab: The Matlab reader object containing matched chains and names.
        :type matlab: MatlabReader
        :param ratio: The ratio for scaling transformations.
        :param corners: The corners of the montage.
        """
        self.elements = elements
        self.center_point = center_point
        self.base_path = base_path
        self.ratio = ratio
        self.corners = corners
        self.matched_chains = matlab.matched_chains
        self.names = [name[0] for name in matlab.names]
        self.ssim_path = "" # Parser.ssim_path
        self.n = 5 #Parser.n
        self.parallel = True # parser.
        self.ssim_results = {}
        self.computed_pairs = set()
        self.lock = Lock()
        self.Processes = [] if self.parallel else None

    def improve_montage_correction_with_ssim(self):
        """
        Improve montage correction using SSIM.

        This method processes the montage elements to improve their alignment using SSIM,
        joins any parallel processes, and updates the center location.

        :return: The updated center location.
        :rtype: int
        """
        logging.info("Starting the SSIM correction")
        self.process_elements()
        if len(self.ssim_results) < 1:
            return 0
        self.join_processes()
        self.process_ssim_results()
        center_loc = self.update_center_location()
        #TODO: call manager to handle directories
        # Write the output images updated through the SSIM
        # directory_before = os.path.join(self.ssim_path, "before_ssim")
        # os.makedirs(directory_before, exist_ok=True)
        # files = os.listdir(self.corrected_path)
        # deplace_files(self.corrected_path, directory_before, files)
        # self.write_images()

        # save_checkpoint(self.corrected_path,
        #                 {"shifts": self.shifts},
        #                 {"center": self.center},
        #                 {"tot_shape": self.tot_shape},
        #                 {"saved_images": self.saved_images})

    def process_elements(self):
        """
        Process all montage elements.

        This method processes each montage element and its neighbors to compute SSIM scores.
        """
        for element in self.elements:
            neighbors = self.get_element_neighbors(element)
            self.process_element(element, neighbors)

    def process_element(self, element: MontageElement, neighbors: List[MontageElement]):
        """
        Process a single montage element and its neighbors.

        This method computes the SSIM score for the given element and its neighbors if they have an overlapping region.

        :param element: The montage element to process.
        :type element: MontageElement
        :param neighbors: The neighboring montage elements.
        :type neighbors: List[MontageElement]
        """
        for neighbor in neighbors:
            overlap_region = element.transform.compute_overlap_region(neighbor.transform)
            if overlap_region.area > 1:
                ssim_score = self.run_ssim(element, neighbor)
                if ssim_score is not None:
                    self.update_ssim_results(element, neighbor, ssim_score)

    def join_processes(self):
        """
        Join parallel processes.

        This method joins all parallel processes if parallel processing is enabled.
        """
        if self.parallel:
            for process in self.Processes:
                process.join()

    def process_ssim_results(self):
        """
        Process SSIM results.

        This method processes the SSIM results by reshaping the SSIM scores and updating the results dictionary.

        :return: None
        """
        for (element, neighbor), ssim_score in self.ssim_results.items():
            ssim_score = np.array(ssim_score).reshape(2*self.n+1, 2*self.n+1)
            self.ssim_results[(element, neighbor)] = ssim_score[:]
            if not np.any(ssim_score):
                continue
            # self.write_ssim_results(ssim_score, element, neighbor, self.ssim_path)

    def update_center_location(self):
        """
        Update the center location based on SSIM results.

        This method updates the transforms and center location of elements based on the SSIM scores.

        :return: None
        """
        # center_loc = self.get_center_loc()
        for (element, neighbor), ssim_score in self.ssim_results.items():
            indice = self.get_indice_max_ssim_score(ssim_score)
            self.update_transforms_and_center(element, neighbor, indice)

    def get_element_neighbors(self, element: MontageElement) -> List[MontageElement]:
        """
        Get neighbors of a montage element.

        This method retrieves the relevant neighbors of a given montage element.

        :param element: The montage element to find neighbors for.
        :type element: MontageElement
        :return: A list of neighboring montage elements.
        :rtype: List[MontageElement]
        """
        index = self.names.index(str(element.image_file))
        chain_index = self.get_key_for_value(index)
        neighbors_image = get_element_neighbors(element, self.elements)
        return self.get_relevant_neighbors(element, neighbors_image, chain_index)

    def get_relevant_neighbors(self, element: MontageElement, neighbors_image: List[ImageFile], chain_index: int) -> List[MontageElement]:
        """
        Get relevant neighbors of a montage element.

        This method filters the neighbors of a given montage element to find the relevant ones based on chain index.

        :param element: The montage element to find neighbors for.
        :type element: MontageElement
        :param neighbors_image: List of neighboring image files.
        :type neighbors_image: List[ImageFile]
        :param chain_index: The chain index of the element.
        :type chain_index: int
        :return: A list of relevant neighboring montage elements.
        :rtype: List[MontageElement]
        """
        neighbors = []
        for neighbor in neighbors_image:
            pair = frozenset({str(neighbor.image_file), str(element.image_file)})
            if pair in self.computed_pairs:
                continue
            self.computed_pairs.add(pair)
            neig_idx = self.names.index(str(neighbor.image_file))
            neig_chain_index = self.get_key_for_value(neig_idx)
            if neig_chain_index != chain_index:
                neighbors.append(neighbor)
        if len(neighbors) > 0:
            print("oe")
        return neighbors

    def run_ssim(self, element: MontageElement, neighbor: MontageElement) -> np.ndarray:
        """
        Run SSIM between two montage elements.

        This method computes the SSIM score between a given element and its neighbor.

        :param element: The montage element to process.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :type neighbor: MontageElement
        :return: The SSIM score as a numpy array.
        :rtype: np.ndarray
        """
        self.prepare_element(element)
        self.prepare_element(neighbor)
        if self.parallel:
            ssim_score = self.run_parallel_ssim(element, neighbor)
        else:
            ssim_score = self.run_sequential_ssim(element, neighbor)
            # ssim_score = list(ssim_score)
        return ssim_score

    def prepare_element(self, element: MontageElement):
        """
        Prepare a montage element for SSIM processing.

        This method reads the data for the given element and prepares it for SSIM processing.

        :param element: The montage element to prepare.
        :type element: MontageElement
        :return: None
        """
        element.image_file.read_data(self.base_path)
        # from src.cell.affine_transform import adapt_transform_for_new_size
        # matrix = adapt_transform_for_new_size(element.transform.matrix)
        # element.adapt_transform_for_new_size
        element.get_ratio_transformed_data(self.ratio, self.corners)
        # element.transformed_data
        element.get_mask_data(element.image_file.data, self.corners)

    def run_parallel_ssim(self, element: MontageElement, neighbor: MontageElement) -> np.ndarray:
        """
        Run SSIM in parallel between two montage elements.

        This method computes the SSIM score between a given element and its neighbor in parallel using multiprocessing.

        :param element: The montage element to process.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :type neighbor: MontageElement
        :return: The SSIM score as a numpy array.
        :rtype: np.ndarray
        """
        self.manage_processes()
        with Manager() as manager:
            manager_list = [0 for _ in range((2*self.n+1)**2)]
            ssim_score = manager.list(manager_list)
            error_queue = manager.Queue()
            p = Process(target=run, args=(element, neighbor, ssim_score, self.ssim_path, self.lock, error_queue))
            self.Processes.append(p)
            p.start()
            p.join()
            if not error_queue.empty():
                error = error_queue.get()
                print(f"Error occurred in subprocess: {error}")
                return None
            return list(ssim_score)

    def run_sequential_ssim(self, element: MontageElement, neighbor: MontageElement) -> np.ndarray:
        """
        Run SSIM sequentially between two montage elements.

        This method computes the SSIM score between a given element and its neighbor sequentially.

        :param element: The montage element to process.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :type neighbor: MontageElement
        :return: The SSIM score as a numpy array.
        :rtype: np.ndarray
        """
        ssim_class = SSIM(element, neighbor, self.ssim_path, self.n)
        try:
            ssim_class.run()
            ssim_score = ssim_class.m
        except ValueError:
            return None
        if not np.any(ssim_score):
            return None
        # ssim_class.write_ssim_results()
        return ssim_score

    def manage_processes(self):
        """
        Manage the number of running processes.

        This method ensures that the number of running processes does not exceed a certain limit.

        :return: None
        """
        # TODO: why 6
        too_many_processes_running = True
        while too_many_processes_running:
            number_of_processes_running = len([process for process in self.Processes if process.is_alive()])
            if number_of_processes_running < 6:
                too_many_processes_running = False
            else:
                time.sleep(600)

    def update_ssim_results(self, element: MontageElement, neighbor: MontageElement, ssim_score: np.ndarray):
        """
        Update SSIM results.

        This method updates the SSIM results dictionary with the computed SSIM score for a given element and its neighbor.

        :param element: The montage element.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :type neighbor: MontageElement
        :param ssim_score: The computed SSIM score.
        :type ssim_score: np.ndarray
        :return: None
        """
        with self.lock:
            self.ssim_results[(element, neighbor)] = ssim_score

    @staticmethod
    def get_indice_max_ssim_score(ssim_score: Union[np.ndarray, List[float]]) -> List[int]:
        """
        Get the indices of the maximum SSIM score.

        This method computes the indices of the maximum SSIM score in the given SSIM score array.

        :param ssim_score: The SSIM score array.
        :type ssim_score: Union[np.ndarray, List[float]]
        :return: The indices of the maximum SSIM score.
        :rtype: List[int]
        """
        try:
            indice = np.unravel_index(ssim_score.argmax(), ssim_score.shape)
        except AttributeError:
            ssim_score = np.array(ssim_score).reshape(int(np.sqrt(len(ssim_score))), int(np.sqrt(len(ssim_score))))
            indice = np.unravel_index(ssim_score.argmax(), ssim_score.shape)
        indice = [indice[0]-math.floor(ssim_score.shape[0]/2), indice[1] - math.floor(ssim_score.shape[1]/2)]
        return indice

    def update_transforms_and_center(self, element: MontageElement, neighbor: MontageElement, indice: List[int]):
        """
        Update transforms and center location.

        This method updates the transforms and center location of elements based on the SSIM score indices.

        :param element: The montage element.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :type neighbor: MontageElement
        :param indice: The indices of the maximum SSIM score.
        :type indice: List[int]
        :return: None
        """
        if self.is_the_element_to_move(element, neighbor):
            self.increase_neighbors(neighbor, indice)
            self.increase_center(neighbor, indice)
        else:
            self.increase_neighbors(element, indice)
            self.increase_center(element, indice)

    def increase_center(self, element: MontageElement, indice: List[int]):
        """
        Increase the center location.

        This method increases the center location based on the SSIM score indices.

        :param element: The montage element.
        :type element: MontageElement
        :param indice: The indices of the maximum SSIM score.
        :type indice: List[int]
        :return: None
        """
        if element.transformed_mask[self.center_point.y, self.center_point.x] == 1:
            return
        elif element.image_file.x_position == 0 and self.center_point.y > element.image_file.y_position:
            self.center_point.x += indice[1]
            self.center_point.y += indice[0]
        elif element.image_file.y_position == 0 and self.center_point.x > element.image_file.x_position:
            self.center_point.x += indice[1]
            self.center_point.y += indice[0]

    def increase_neighbors(self, element: MontageElement, indice: List[int]):
        """
        Increase the transforms of neighboring elements.

        This method increases the transforms of neighboring elements based on the SSIM score indices.

        :param element: The montage element.
        :type element: MontageElement
        :param indice: The indices of the maximum SSIM score.
        :type indice: List[int]
        :return: None
        """
        for neighbor_element in self.elements:
            if self.is_the_element_to_move_x(neighbor_element, element):
                neighbor_element.transform.matrix[0, 2] += indice[1]
            if self.is_the_element_to_move_y(neighbor_element, element):
                neighbor_element.transform.matrix[1, 2] += indice[0]

    def is_the_element_to_move_x(self, element: MontageElement, neighbor: MontageElement) -> bool:
        """
        Check if the element should be moved in the x direction.

        This method checks if the given element should be moved in the x direction based on its neighbor.

        :param element: The montage element.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :type neighbor: MontageElement
        :return: True if the element should be moved in the x direction, False otherwise.
        :rtype: bool
        """
        return True if element.transform.is_right(neighbor.transform) else False

    def is_the_element_to_move_y(self, element: MontageElement, neighbor: MontageElement) -> bool:
        """
        Check if the element should be moved in the y direction.

        This method checks if the given element should be moved in the y direction based on its neighbor.

        :param element: The montage element.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :return: True if the element should be moved in the y direction, False otherwise.
        :rtype: bool
        """
        return True if element.transform.is_below(neighbor.transform) else False

    def is_the_element_to_move(self, element: MontageElement, neighbor: MontageElement) -> bool:
        """
        Check if the element should be moved.

        This method checks if the given element should be moved based on its neighbor.

        :param element: The montage element.
        :type element: MontageElement
        :param neighbor: The neighboring montage element.
        :type neighbor: MontageElement
        :return: True if the element should be moved, False otherwise.
        :rtype: bool
        """
        if element.image_file.x_position == neighbor.image_file.x_position:
            return True if element.transform.is_below(neighbor.transform) else False
        elif element.image_file.y_position == neighbor.image_file.y_position:
            return True if element.transform.is_right(neighbor.transform) else False
        else:
            print("They are not neighbors")
            raise RuntimeError

    # def get_center_loc(self) -> None:
    #     """
    #     Find where the center of the image is to also add shift from SSIM
    #     """

    #     middle_shifts = {}
    #     for _, shift in self.shifts.iterrows():
    #         middle_shifts[shift["name"]] = [(shift["top_left_y"]+shift["bottom_right_y"])/2,
    #                                         (shift["top_left_x"]+shift["bottom_right_x"])/2]
    #     # Look for the smallest distance from center to image and find in which
    #     # the center is
    #     closest_dist = np.inf
    #     closest = None
    #     for image_name, center_shift in middle_shifts.items():
    #         dist = np.sqrt(
    #             np.square(self.center[0]-center_shift[0]) + np.square(self.center[1]-center_shift[1])
    #         )
    #         if dist < closest_dist:
    #             closest_dist = dist
    #             closest = image_name
    #     center_loc = eval(re.search(r"\((.*?)\)", closest).group(0))

    #     return center_loc

    def get_key_for_value(self, value):
        for k, v in self.matched_chains.items():
            if value in v:
                return k
        return None

def run(element, neighbor, ssim_score, ssim_path, lock, error_queue) -> np.ndarray:
    """
    Function to get a matrices of the SSIM score around a neighborhood of n pixels
    with help of parallelization

    :param img1: the first image
    :type img1: np.ndarray
    :param name1: the first image name
    :type name1: str
    :param img2: the second image
    :type img2: np.ndarray
    :param name2: the second image name
    :type name2: str
    :param mask: the mask where ones is the area of overlap between the 2 images
    :type mask: np.ndarray
    :param ssim_matrice: the score of each pixel's SSIM
    :type ssim_matrice: np.ndarray
    :param ssim_path: the output path where results of SSIM will be
    :type ssim_path: str
    :return: the score of each pixel's SSIM updated
    :rtype: np.ndarray
    """

    logging.info("Run SSIM in parallel")
    # get the number of pixel we want to search through
    try:
        n = int(np.sqrt(len(ssim_score))/2)
        k = 0
        ssim_class = SSIM(element, neighbor, ssim_path, n)
        for i in range(-n, n+1):
            for j in range(-n, n+1):
                # Start a parallel process that runs the SSIM
                score = ssim_class.run_ssim(i, j)

                # Use lock to synchronize access to the shared ssim_score list
                with lock:
                    ssim_score[k] = score
                k += 1
        # ssim_class.write_ssim_results(ssim_matrice)
        return ssim_score
    except ValueError as e:
        error_queue.put(e)