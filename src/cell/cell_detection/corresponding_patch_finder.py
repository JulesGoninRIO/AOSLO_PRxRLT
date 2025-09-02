from typing import Dict, List, Tuple
import logging
import io
import cv2
import os
import re
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.cell.montage.montage_element import MontageElement
from src.cell.montage.montage_mosaic import MontageMosaic
from src.shared.datafile.image_file import ImageFile
from src.shared.datafile.coordinates_file import CoordinatesFile

def calculate_cross_image_distance(patch1: CoordinatesFile, patch2: str, inter_image_distance: float) -> float:
    """
    Calculate the cross-image distance between two patches.

    This function calculates the distance between two patches, considering their positions
    and the inter-image distance. It handles cases where the patches are aligned either
    horizontally, vertically, or diagonally.

    :param patch1: The first patch with coordinates.
    :type patch1: CoordinatesFile
    :param patch2: The second patch with coordinates.
    :type patch2: str
    :param inter_image_distance: The distance between images.
    :type inter_image_distance: float
    :return: The calculated distance between the two patches.
    :rtype: float
    """
    if patch1.x_position == patch2.x_position:
        if patch1.y_position > patch2.y_position:
            return inter_image_distance - patch1.y_position + patch2.y_position
        else:
            return inter_image_distance + patch1.y_position - patch2.y_position
    elif patch1.y_position == patch2.y_position:
        if patch1.x_position < patch2.x_position:
            return inter_image_distance + patch1.x_position - patch2.x_position
        else:
            return inter_image_distance - patch1.x_position + patch2.x_position
    else:
        # Calculate the diagonal distance
        x_distance = abs(patch1.x_position - patch2.x_position)
        y_distance = abs(patch1.y_position - patch2.y_position)
        diagonal_distance = np.sqrt(x_distance**2 + y_distance**2)
        # if 0 < diagonal_distance < 480:
        #     return diagonal_distance
        # else:
        #     return np.inf()
        return diagonal_distance

VISUALIZE_PATCHES = False
IMAGE_SIZE = 720
PATCH_SIZE = 185

class CorrespondingPatchFinder():
    """
    A class to find corresponding patches in an image mosaic.

    This class processes an image mosaic to find corresponding patches based on the number of cones detected
    and other criteria. It filters patches by a threshold and groups similar patches.

    :param image_mosaic: The image mosaic to process.
    :type image_mosaic: MontageMosaic
    :param output_path: The path to save output data.
    :type output_path: Path
    :param histogram_per_patch: A dictionary containing histograms for each patch.
    :type histogram_per_patch: Dict[ImageFile, int]
    """

    def __init__(self, image_mosaic: MontageMosaic, output_path: Path, histogram_per_patch: Dict[ImageFile, int]):
        self.__visualize_patches = VISUALIZE_PATCHES
        self.__analysis_path = '' # analysis_path
        self.image_mosaic = image_mosaic
        self.output_path = output_path
        self.histogram_per_patch = histogram_per_patch
        self.number_cones_detected_per_patch = get_number_of_cones_per_patch(self.output_path / 'output_cone_detector' / 'algorithmMarkers')
        self.threshold = self.get_threshold()
        self.patches_to_treat = self.filter_by_threshold('both')
        self.corresponding_patches = {}
        self.patch_groups = find_similar_names(list(self.number_cones_detected_per_patch.keys()))
        self.computed_distances = {}
        self.computed_patches = {}

    def filter_by_threshold(self, treatment_scope: str) -> Dict[str, int]:
        """
        Filter patches by the number of cones detected based on a threshold.

        This method filters the patches based on whether the number of cones detected is above, below,
        or both relative to the threshold.

        :param treatment_scope: The scope of filtering ('above', 'below', or 'both').
        :type treatment_scope: str
        :return: A dictionary of filtered patches with their cone counts.
        :rtype: Dict[str, int]
        :raises ValueError: If the treatment_scope is not 'above', 'below', or 'both'.
        """
        if treatment_scope not in ['above', 'below', 'both']:
            raise ValueError("Condition must be 'above', 'below', or 'both'")
        if treatment_scope == 'above':
            return {key: value for key, value in self.number_cones_detected_per_patch.items() if value > self.threshold}
        elif treatment_scope == 'below':
            return {key: value for key, value in self.number_cones_detected_per_patch.items() if value < self.threshold}
        elif treatment_scope == 'both':
            return self.number_cones_detected_per_patch

    def get_threshold(self):
        """
        Calculate the threshold for filtering patches based on the number of cones detected.

        This method calculates the threshold using Otsu's method on the histogram of the number of cones detected.

        :return: The calculated threshold.
        :rtype: float
        """
        logging.info("Analyzing cones detected..")
        cones_detected_hist = create_histogram_distribution(np.array(list(self.number_cones_detected_per_patch.values())))
        threshold = get_otsu_threshold(cones_detected_hist)

        # #TODO: create the plots
        # # create necessary things to plot
        # PreprocessingPlotter().plot_otsu(cones_detected_hist, threshold, self.analysis_path)
        # # separated_patches = separate_patches_based_on_threshold(number_cones_detected_per_patch, threshold)

        # # Creates all the plots of the analyis
        # """ pixels_before ??? """
        # PreprocessingPlotter.preprocessing_plots(self.base_dir, self.analysis_path, pixels, cones_detected_hist)


        # # We want to have the occurence of pixel values per patch
        # if self.previous_dir:
        #     output_path = self.previous_path
        # else:
        #     output_path = self.output_path
        # patch_dir = os.path.join(output_path, RAW_DATA_DIR_NAME)

        # # Save the analysis results in the specific folder
        # if self.analysis_path:
        #     logging.info(f"Cleaning folder {self.analysis_path} to save analysis outputs.")
        #     if os.path.isdir(self.analysis_path):
        #         empty_folder(self.analysis_path)
        #     os.makedirs(self.analysis_path, exist_ok=True)


        # # # renames the folder we have taken the images from
        # # if name :
        # #     out_name = os.path.join(Path(base_dir).parent.absolute(), name)
        # #     if os.path.isdir(out_name):
        # #         shutil.rmtree(out_name)
        # #     os.rename(base_dir, out_name)
        return threshold

    def get_corresponding_patches(self) -> Dict[CoordinatesFile, CoordinatesFile]:
        """
        Get corresponding patches for all patches to treat.

        This method iterates over all patches to treat and finds their corresponding patches.

        :return: A dictionary mapping each patch to treat to its corresponding patch.
        :rtype: Dict[CoordinatesFile, CoordinatesFile]
        """
        for patch_to_treat in self.patches_to_treat:
            self.get_corresponding_patch(patch_to_treat)
        return self.corresponding_patches

    def get_corresponding_patch(self, patch_to_treat: CoordinatesFile):
        """
        Get the corresponding patch for a single patch to treat.

        This method finds the corresponding patch for a given patch to treat by calculating distances
        and processing neighbors.

        :param patch_to_treat: The patch to find a corresponding patch for.
        :type patch_to_treat: CoordinatesFile
        """
        try:
            patch_to_treat = CoordinatesFile(patch_to_treat)
            image_coordinates = patch_to_treat.without_prefix()
            image = image_coordinates.to_image_file()
        except ValueError:
            patch_to_treat = ImageFile(patch_to_treat)
            image = patch_to_treat.without_prefix()
            image_coordinates = image
        other_patches_same_image = self.patch_groups[str(image_coordinates)]
        distances = find_distance_to_all_patches(patch_to_treat, other_patches_same_image)
        try:
            element = self.get_element_from_image_mosaic(image)
            if element.transform is None:
                self.image_mosaic.set_transforms()
            neig_patch_distance = self.get_neighbor_patch_distance(element)
            self.process_neighbors(neig_patch_distance, patch_to_treat, distances)
        except ValueError:
            # no neighbor in the mosaic
            self.update_best_neighbor(distances, patch_to_treat)

    def get_corr_patch(self, corrected_path, pixels):
        """
        Get the corresponding working patch to a non-working patch

        :param out_dir: the directory where the files are
        :param montage_dir: the directory where the montage files are
        :param analysis_dir: the directory where to save plots
        :param separated_patches: the patches separated
        :param cones_detected: the number of cones detected per patch
        :param treat: whether we treat working, non-working or both patches
        :param visualize: whether we want to visualize the output
        :param solo: whether we want to correct with only one patch or multiple
        """

        separated_patches, cones_detected = self.get_statistics(pixels)
        shifts = create_own_shifts(corrected_path)
        neighbors_image = subject_neighbor(corrected_path, shifts)
        neig_patch_dic = get_neighbor_patch_distance(self.previous_path, neighbors_image)

        # Here we want to correct images based on relative locations
        corr_patch = {}
        incurrable_patches = []
        distances = []
        num_cones = []
        if self.__treats == "both" :
            treats = ["good", "bad"]
        else :
            treats = [self.__treats]
        for treat_method in treats :
            for patch in separated_patches[treat_method] :
                if self.solo :
                    best_count = 0
                    best_dist = 0
                    best_num_cones = 0
                    best_val = None
                    for potential_ref in neig_patch_dic[patch] :
                        # count = cones_detected[potential_ref[0]] / np.log10(potential_ref[1])
                        # count = cones_detected[potential_ref[0]] *(1-(1/500)*(potential_ref[1]))
                        count = cones_detected[potential_ref[0]] *(1-(1/1000)*(potential_ref[1]))
                        if count > best_count:
                            best_count = count
                            best_dist = potential_ref[1]
                            best_num_cones = cones_detected[potential_ref[0]]
                            best_val = potential_ref[0]
                    if best_val :
                        corr_patch[patch] = best_val
                        distances.append(best_dist)
                        num_cones.append(best_num_cones)
                    else :
                        incurrable_patches.append(patch)
                        logging.info(f"{patch} is incurrable")
                        # print(f"{patch} is incurrable")
                else :
                    best_vals = [val for val in neig_patch_dic[patch] if neig_patch_dic[patch] in separated_patches["good"]]
                    if best_vals :
                        corr_patch[patch] = best_vals

        # plt.hist(distances)
        # plt.xlabel("Distance in pixels")
        # plt.ylabel("Occurence")
        # plt.title("Distance of the styled patch")
        # plt.savefig(os.path.join(self.analysis_path, "dist_log.png"))
        # plt.close("all")

        # plt.hist(num_cones)
        # plt.xlabel("Number of cones detected")
        # plt.ylabel("Occurence")
        # plt.title("Number of cones in the styled patch")
        # plt.savefig(os.path.join(self.analysis_path, "num_cones_log.png"))

        # Analysis of positions taken to correct patches
        count_same = 0
        count_diff = 0
        for key, val in corr_patch.items():
            key_img = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", key)
            val_img = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", val)
            if val_img == key_img :
                count_same+=1
            else :
                count_diff+=1
        logging.info(f"Number of patch taken on same image: {count_same}, on different image: {count_diff}")

    def get_element_from_image_mosaic(self, image: ImageFile) -> MontageElement:
        """
        Get the element from the image mosaic.

        This method retrieves the element corresponding to the given image from the image mosaic.

        :param image: The image to get the element for.
        :type image: ImageFile
        :return: The corresponding montage element.
        :rtype: MontageElement
        :raises ValueError: If the element is not found in the image mosaic.
        """
        try:
            return self.image_mosaic.get_element(image)
        except ValueError:
            raise

    def get_neighbor_patch_distance(self, element: MontageElement) -> Dict[CoordinatesFile, float]:
        """
        Get the distances to neighboring patches.

        This method retrieves the distances to neighboring patches for a given element.

        :param element: The element to get neighbor distances for.
        :type element: MontageElement
        :return: A dictionary mapping neighboring patches to their distances.
        :rtype: Dict[CoordinatesFile, float]
        """
        if element in self.computed_distances:
            return self.computed_distances[element]
        else:
            neig_patch_distance = self.image_mosaic.get_element_neighbors_with_distance(element)
            self.computed_distances[element] = neig_patch_distance
            return neig_patch_distance

    def process_neighbors(self, neig_patch_distance: Dict[ImageFile, int], patch_to_treat: ImageFile, distances: Dict[ImageFile, float]):
        """
        Process neighboring patches.

        This method processes neighboring patches by calculating cross-image distances and updating the best neighbor.

        :param neig_patch_distance: The distances to neighboring patches.
        :type neig_patch_distance: Dict[ImageFile, int]
        :param patch_to_treat: The patch to find a corresponding patch for.
        :type patch_to_treat: ImageFile
        :param distances: The distances to all patches.
        :type distances: Dict[ImageFile, float]
        """
        for neighbor, distance in neig_patch_distance.items():
            other_patches = self.computed_patches.get(neighbor)
            if other_patches is None:
                other_patches = self.find_other_patches(neighbor)
                self.computed_patches[neighbor] = other_patches  # Store computed patches
            self.calculate_cross_image_distances(other_patches, patch_to_treat, distance, distances)

    def find_other_patches(self, neighbor: ImageFile) -> List[ImageFile]:
        """
        Find other patches for a given neighbor.

        This method finds other patches that are similar to the given neighbor.

        :param neighbor: The neighbor to find other patches for.
        :type neighbor: ImageFile
        :return: A list of other patches.
        :rtype: List[ImageFile]
        """
        for key in self.patch_groups.keys():
            try:
                if neighbor.is_same_except_modality(CoordinatesFile(key).to_image_file()):
                    return self.patch_groups[key]
            except ValueError:
                if neighbor.is_same_except_modality(ImageFile(key)):
                    return self.patch_groups[key]
        return None

    def calculate_cross_image_distances(self, other_patches: List[ImageFile], patch_to_treat: ImageFile, distance: float, distances: Dict[ImageFile, float]):
        """
        Calculate cross-image distances.

        This method calculates the cross-image distances between the patch to treat and other patches.

        :param other_patches: The other patches to calculate distances to.
        :type other_patches: List[ImageFile]
        :param patch_to_treat: The patch to find a corresponding patch for.
        :type patch_to_treat: ImageFile
        :param distance: The distance to the neighboring patch.
        :type distance: float
        :param distances: The distances to all patches.
        :type distances: Dict[ImageFile, float]
        """
        cross_image_distances = {}
        for other_patch in other_patches:
            cross_image_distances[other_patch] = calculate_cross_image_distance(patch_to_treat, other_patches[0], distance)
        distances.update(cross_image_distances)
        self.update_best_neighbor(distances, patch_to_treat)

    def update_best_neighbor(self, distances: Dict[ImageFile, float], patch_to_treat: ImageFile):
        """
        Update the best neighbor for a patch to treat.

        This method updates the best neighbor for a given patch to treat based on the calculated distances.

        :param distances: The distances to all patches.
        :type distances: Dict[ImageFile, float]
        :param patch_to_treat: The patch to find a corresponding patch for.
        :type patch_to_treat: ImageFile
        """
        best_patch = self.find_best_neighbor(distances, self.number_cones_detected_per_patch)
        self.corresponding_patches[patch_to_treat] = best_patch

    @staticmethod
    def find_best_neighbor(distances: Dict[ImageFile, float], number_cones_detected_per_patch: Dict[str, int]) -> ImageFile:
        """
        Find the best neighboring patch.

        This method finds the best neighboring patch based on the distances and the number of cones detected.

        :param distances: A dictionary mapping patches to their distances.
        :type distances: Dict[ImageFile, float]
        :param number_cones_detected_per_patch: A dictionary mapping patches to the number of cones detected.
        :type number_cones_detected_per_patch: Dict[str, int]
        :return: The best neighboring patch.
        :rtype: ImageFile
        """
        best_score = 0
        best_dist = 0
        best_num_cones = 0
        best_val = None
        for potential_best_neig, distance in distances.items():
            number_of_cones = number_cones_detected_per_patch[str(potential_best_neig)]
            score = CorrespondingPatchFinder.calculate_best_count(distance, number_of_cones)
            if score > best_score:
                best_score, best_dist, best_num_cones, best_val = score, distance, number_of_cones, potential_best_neig
        if best_val:
            return best_val
        else:
            return None

    @staticmethod
    def calculate_best_count(distance: float, number_of_cones_detected: int) -> float:
        """
        Calculate the best count score.

        This method calculates the best count score based on the distance and the number of cones detected.

        :param distance: The distance to the neighboring patch.
        :type distance: float
        :param number_of_cones_detected: The number of cones detected in the patch.
        :type number_of_cones_detected: int
        :return: The best count score.
        :rtype: float
        """
        slope = 1/1000 # worse with 1/500
        return number_of_cones_detected * (1 - slope * distance)

    def plot_histograms(self, distances: Dict[ImageFile, float], num_cones: Dict[ImageFile, int]):
        """
        Plot histograms of distances and number of cones.

        This method plots histograms of the distances and the number of cones detected in the patches.

        :param distances: A dictionary mapping patches to their distances.
        :type distances: Dict[ImageFile, float]
        :param num_cones: A dictionary mapping patches to the number of cones detected.
        :type num_cones: Dict[ImageFile, int]
        """
        plt.hist(distances)
        plt.xlabel("Distance in pixels")
        plt.ylabel("Occurrence")
        plt.title("Distance of the styled patch")
        plt.savefig(os.path.join(self.analysis_path, "dist_log.png"))
        plt.close("all")

        plt.hist(num_cones)
        plt.xlabel("Number of cones detected")
        plt.ylabel("Occurrence")
        plt.title("Number of cones in the styled patch")
        plt.savefig(os.path.join(self.analysis_path, "num_cones_log.png"))

    def analyze_positions(self, corr_patch: Dict[str, str]) -> Tuple[int, int]:
        """
        Analyze the positions of corrected patches.

        This method analyzes the positions of the corrected patches and counts how many are from the same image
        and how many are from different images.

        :param corr_patch: A dictionary mapping patches to their corrected patches.
        :type corr_patch: Dict[str, str]
        :return: A tuple containing the count of patches from the same image and the count of patches from different images.
        :rtype: Tuple[int, int]
        """
        count_same = 0
        count_diff = 0
        for key, val in corr_patch.items():
            key_img = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", key)
            val_img = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", val)
            if val_img == key_img:
                count_same += 1
            else:
                count_diff += 1
        logging.info(f"Number of patch taken on same image: {count_same}, on different image: {count_diff}")
        return count_same, count_diff

    def visualize_patches(self, neig_patch_dic: Dict[str, str], corr_patch: Dict[str, str]):
        """
        Visualize the patches.

        This method visualizes the patches by drawing rectangles around the patches and their corresponding patches.

        :param neig_patch_dic: A dictionary mapping patches to their neighboring patches.
        :type neig_patch_dic: Dict[str, str]
        :param corr_patch: A dictionary mapping patches to their corrected patches.
        :type corr_patch: Dict[str, str]
        """
        if self.__visualize_patches:
            for key, vals in neig_patch_dic.items():
                session = re.search(r'Session(\d+)', key).group(1)
                session_dir = os.path.join(os.path.join(r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1", "Session"+session), "corrected")
                image_name = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", key)
                image_name = re.sub("CalculatedSplit", "Confocal", image_name)
                image = cv2.imread(os.path.join(session_dir, image_name), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                meaningful_mask = image > 0
                meaningful_idx = np.argwhere(meaningful_mask)

                top = np.min(meaningful_idx[:, 0])
                bottom = np.max(meaningful_idx[:, 0])
                left = np.min(meaningful_idx[:, 1])
                right = np.max(meaningful_idx[:, 1])

                xy_position = (re.search(r"_x(\d+)y(\d+)", key).group(1), re.search(r"_x(\d+)y(\d+)", key).group(2))
                top_left = (left+int(xy_position[0]), top+int(xy_position[1]))
                bottom_right = (top_left[0] + 185, top_left[1]+185)
                # The actual patch in red
                image = cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), thickness=-1)

                val_image_loaded = [image_name]
                # color in green the best patch
                for val in vals:
                    crop_name = val[0]
                    val_image_name = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", crop_name)
                    val_image_name = re.sub("CalculatedSplit", "Confocal", val_image_name)
                    if val_image_name == val_image_loaded[0]:
                        # no need to reload the big image, only draw the rectangle on it
                        val_xy_position = (re.search(r"_x(\d+)y(\d+)", crop_name).group(1), re.search(r"_x(\d+)y(\d+)", crop_name).group(2))
                        val_top_left = (left+int(val_xy_position[0]), top+int(val_xy_position[1]))
                        val_bottom_right = (val_top_left[0] + 185, val_top_left[1]+185)
                        image = cv2.rectangle(image, val_top_left, val_bottom_right, (255, 0, 0))
                    else:
                        if val_image_name not in val_image_loaded:
                            val_image_loaded.append(val_image_name)
                            val_image = cv2.imread(os.path.join(session_dir, val_image_name), cv2.IMREAD_COLOR)
                            meaningful_mask = val_image > 0
                            meaningful_idx = np.argwhere(meaningful_mask)

                            val_top = np.min(meaningful_idx[:, 0])
                            val_bottom = np.max(meaningful_idx[:, 0])
                            val_left = np.min(meaningful_idx[:, 1])
                            val_right = np.max(meaningful_idx[:, 1])

                            image = cv2.addWeighted(image, 0.5, val_image, 0.5, 0)

                        xy_position = (re.search(r"_x(\d+)y(\d+)", crop_name).group(1), re.search(r"_x(\d+)y(\d+)", crop_name).group(2))
                        top_left = (val_left+int(xy_position[0]), val_top+int(xy_position[1]))
                        bottom_right = (top_left[0] + 185, top_left[1]+185)
                        # The actual patch in red
                        image = cv2.rectangle(image, top_left, bottom_right, (255, 0, 0))

                # Draw best image in green
                if key in corr_patch.keys():
                    best_val = corr_patch[key]
                    val_image_name = re.sub(r"CROP_(\d+)_x(\d+)y(\d+)_", "", best_val)
                    val_image_name = re.sub("CalculatedSplit", "Confocal", val_image_name)
                    val_xy_position = (re.search(r"_x(\d+)y(\d+)", best_val).group(1), re.search(r"_x(\d+)y(\d+)", best_val).group(2))
                    val_top_left = (left+int(val_xy_position[0]), top+int(val_xy_position[1]))
                    val_bottom_right = (val_top_left[0] + 185, val_top_left[1]+185)
                    image = cv2.rectangle(image, val_top_left, val_bottom_right, (0, 255, 0), thickness=3)

                    cv2.imwrite(os.path.join(self.analysis_path, key), image)

def create_histogram_distribution(num_cones_per_patch_values: np.array) -> Tuple[np.array, np.array]:
    """
    Create a histogram distribution of the number of cones per patch.

    This method creates a histogram distribution of the number of cones detected per patch.

    :param num_cones_per_patch_values: An array containing the number of cones detected per patch.
    :type num_cones_per_patch_values: np.array
    :return: A tuple containing the histogram of cones detected and the bin edges.
    :rtype: Tuple[np.array, np.array]
    """
    max_of_cones_per_patch = np.max(num_cones_per_patch_values)
    cones_detected_hist = np.histogram(
        num_cones_per_patch_values,
        max_of_cones_per_patch,
        [0, max_of_cones_per_patch]
    )
    return cones_detected_hist

def get_otsu_threshold(cones_detected_hist: Tuple[np.array, np.array]) -> float:
    """
    Get the OTSU threshold of an histogram

    :param cones_detected_hist: histogram of the number of cones detected
    :type cones_detected_hist: Tuple[np.array, np.array]
    :return: the threshold corresponding to the bins to separate the histogram
    :rtype: float
    """
    counts = cones_detected_hist[0]
    bin_centers = 0.5*(cones_detected_hist[1][1:]+cones_detected_hist[1][:-1])
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    if np.all(weight2 == 0):
        return 0
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold

def pixels_per_patch(patch_dir: str) -> Dict[str, np.array]:
    """
    Function to get the pixel values of every patches used

    :param patch_dir: The patch directory
    :type patch_dir: str
    :return: a dictionary with all the pixel values that are in the patch
    :rtype: Dict[str, np.array]
    """
    # look for the patches used to run Davidson's algorithm
    # output dir is the base directory of output -> look for output from
    # Davidson's algorithm

    pixels = {}

    # number_of_entries = len(os.listdir(patch_dir))
    # logging.info("Reading cone detection output")
    for file in os.scandir(patch_dir):
        if os.path.isfile(os.path.join(patch_dir, file)):
            if file.name.endswith(".tif"):
                # get histogram of single patch image
                image = ImageFile(file.name)
                image.read_data(patch_dir)
                hist = np.histogram(image.data, 256, [0, 256])
                pixels[file.name] = hist

    return pixels

def get_number_of_cones_per_patch(cone_path: Path) -> Dict[str, int]:
    """
    Get the number of cones per patch.

    This method scans the given directory for files and calculates the number of cones detected in each patch.

    :param cone_path: The path to the directory containing the cone files.
    :type cone_path: Path
    :return: A dictionary mapping patch names to the number of cones detected.
    :rtype: Dict[str, int]
    """
    num_cones_per_patch = {}
    for file in os.scandir(cone_path):
        num_cones_per_patch[str(CoordinatesFile(file.name).to_image_file())] = get_number_of_cones(file)
    return num_cones_per_patch

def get_number_of_cones(filepath: Path) -> int:
    """
    Get the number of cones from a file.

    This method reads the given file and counts the number of lines, which corresponds to the number of cones detected.

    :param filepath: The path to the file containing cone data.
    :type filepath: Path
    :return: The number of cones detected.
    :rtype: int
    """
    with io.open(filepath, 'r', encoding='utf8') as f:
        lines = len(f.readlines())
    return lines

def find_similar_names(file_list: List[str]) -> Dict[str, List[CoordinatesFile]]:
    """
    Find similar names in a list of files.

    This method removes the "CROP_n_xn_yn_" part from each string and groups the results in a dictionary.

    :param file_list: A list of file names.
    :type file_list: List[str]
    :return: A dictionary mapping stripped names to a list of CoordinatesFile objects.
    :rtype: Dict[str, List[CoordinatesFile]]
    """
    stripped_names = {}
    for name in file_list:
        try:
            coordinate_name = CoordinatesFile(name)
        except ValueError:
            coordinate_name = ImageFile(name)
        if coordinate_name.is_patch():
            stripped_name = str(coordinate_name.without_prefix())
            if stripped_name in stripped_names:
                stripped_names[stripped_name].append(coordinate_name)
            else:
                stripped_names[stripped_name] = [coordinate_name]
    return stripped_names

def calculate_distance(patch1: CoordinatesFile, patch2: CoordinatesFile) -> Tuple[int, int]:
    """
    Calculate the distance between two patches.

    This method calculates the Euclidean distance between two patches based on their coordinates.

    :param patch1: The first patch.
    :type patch1: CoordinatesFile
    :param patch2: The second patch.
    :type patch2: CoordinatesFile
    :return: The distance between the two patches.
    :rtype: Tuple[int, int]
    """
    return math.sqrt((patch1.crop_x_position - patch2.crop_x_position)**2 + (patch1.crop_y_position - patch2.crop_y_position)**2)

def find_distance_to_all_patches(patch: CoordinatesFile, patch_list: List[CoordinatesFile]) -> Dict[CoordinatesFile, float]:
    """
    Find the distance to all patches.

    This method calculates the distance from the given patch to all other patches in the list.

    :param patch: The patch to calculate distances from.
    :type patch: CoordinatesFile
    :param patch_list: A list of patches to calculate distances to.
    :type patch_list: List[CoordinatesFile]
    :return: A dictionary mapping patches to their distances from the given patch.
    :rtype: Dict[CoordinatesFile, float]
    """
    distances = {}
    for other_patch in patch_list:
        if other_patch != patch:
            distance = calculate_distance(patch, other_patch)
            distances[other_patch] = distance
    return distances
