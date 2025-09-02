import logging
import os
import pickle
# newest versions of OpenCV changed the way of importing
import shutil
import warnings
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.shared.computer_vision.voronoi import VoronoiDiagram

from src.cell.cell_detection.atms._plotting import plot_3d_distribution
from src.cell.cell_detection.atms._preparation import (get_roi_from_images,
                                    get_roi_from_montaged_images,
                                    get_roi_from_shift_images)
# from .postprocessing import VoronoiDiagram
from src.cell.cell_detection.atms.particle_refinement import (
    ParticleModel, refine_cell_coordinates
)
from pathlib import Path
try:
    from cv2 import cv2
except ImportError:
    import cv2

import numpy as np
from tqdm import tqdm, trange

from src.configs.parser import Parser
from src.shared.datafile.image_file import ImageFile
from src.shared.datafile.coordinates_file import CoordinatesFile
from src.shared.datafile.helpers import ImageModalities
from src.cell.cell_detection.atms._constants import (
    DEGREE_COORDINATES_DATATYPE, GlobalThresholdContainer)
MONTAGE_IMAGE = "Confocal.tif"

from src.cell.cell_detection.atms._constants import (
    AVERAGING_SIZE, AVERAGING_SIZE_IN_UM, AXIAL_LENGTH_FILE,
    EMPTY_DEGREE_COORDINATES, FOVEA_BORDER, OUTER_FOVEA_BORDER, PIXELS_PER_DEGREE,
    RETINA_AREA_PATTERN, ROI_SIZE, UM_PER_DEGREE, UM_PER_PIXEL)
from src.cell.cell_detection.atms._rescaling import convert_global_coordinates_deg2pixel, pixel2deg
from src.cell.cell_detection.atms._plotting import plot_density_histograms, highlight_coordinates_on_image
from src.cell.cell_detection.atms._detection import get_cell_coordinates_on_image
from src.cell.cell_detection.atms._preparation import select_images


def is_detected_well(coordinates: np.ndarray, retina_area: int) -> bool:
    """
    Check if the given coordinates are not empty and cover all the retina.

    This function checks if the provided coordinates array is not empty and if the area covered by the coordinates
    is greater than half of the retina area.

    :param coordinates: The coordinates of detected cells.
    :type coordinates: np.ndarray
    :param retina_area: The total area of the retina.
    :type retina_area: int
    :return: True if the coordinates cover more than half of the retina area, False otherwise.
    :rtype: bool
    """
    if coordinates.size == 0:
        return False

    min_y = int(np.min(coordinates[:, 0]))
    min_x = int(np.min(coordinates[:, 1]))
    max_y = int(np.max(coordinates[:, 0]))
    max_x = int(np.max(coordinates[:, 1]))
    covered_area = (max_x - min_x) * (max_y - min_y)
    return covered_area > retina_area // 2


def get_densities_across_roi(
        cell_coordinates: np.ndarray,
        image: np.ndarray = None,
        bitmap: np.ndarray = None,
        um_per_degree: float = UM_PER_DEGREE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the ROI into several windows and compute the cell density inside each of them.

    This function divides the Region of Interest (ROI) into multiple windows and calculates the cell density
    within each window. It also computes additional metrics such as bright area ratios, hexagon ratios, and potential measurements.

    :param cell_coordinates: (y, x) coordinates of the detected cells in pixels.
    :type cell_coordinates: np.ndarray
    :param image: The image of the ROI, defaults to None.
    :type image: np.ndarray, optional
    :param bitmap: Bright area bitmap of the ROI, defaults to None.
    :type bitmap: np.ndarray, optional
    :param um_per_degree: Micrometers per degree, defaults to UM_PER_DEGREE.
    :type um_per_degree: float, optional
    :return: A tuple containing the positions of window centers, cell counts, cell densities, bright area ratios, hexagon ratios, and potential measurements.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    if bitmap is None:
        bitmap = np.ones((ROI_SIZE, ROI_SIZE), dtype=bool)

    y_number_of_windows = int(np.ceil(bitmap.shape[0] / AVERAGING_SIZE))
    x_number_of_windows = int(np.ceil(bitmap.shape[1] / AVERAGING_SIZE))

    cell_numbers = np.zeros(y_number_of_windows * x_number_of_windows, dtype=np.uint16)
    bright_areas = np.empty(cell_numbers.shape, dtype=np.uint16)
    bright_area_ratios = np.empty_like(bright_areas, dtype=np.float32)
    window_positions = np.empty((y_number_of_windows * x_number_of_windows, 2), dtype=np.uint16)
    hexagon_ratios = np.empty_like(bright_area_ratios)
    potential_measurements = np.empty_like(bright_area_ratios)

    for i in range(y_number_of_windows):
        for j in range(x_number_of_windows):
            k = i * x_number_of_windows + j

            up = i * AVERAGING_SIZE
            left = j * AVERAGING_SIZE
            right = min(left + AVERAGING_SIZE, bitmap.shape[1])
            bottom = min(up + AVERAGING_SIZE, bitmap.shape[0])

            if image is not None:
                window = image[up: bottom, left: right]
            else:
                window = None

            bright_areas[k] = np.count_nonzero(bitmap[up: bottom, left: right])
            bright_area_ratios[k] = float(bright_areas[k]) / (right - left) / (bottom - up)
            if bright_area_ratios[k] < 0.3:
                warnings.warn('Too large part of the image is dark, will zero it out completely', RuntimeWarning)
                cell_numbers[k] = 0
                bright_areas[k] = 0
            else:
                cells_in_window_mask = np.logical_and(
                    np.logical_and(cell_coordinates[:, 0] >= up, cell_coordinates[:, 0] < bottom),
                    np.logical_and(cell_coordinates[:, 1] >= left, cell_coordinates[:, 1] < right)
                )
                cell_numbers[k] = np.count_nonzero(cells_in_window_mask)

                if window is not None:
                    cells_in_window = cell_coordinates[cells_in_window_mask, :]
                    cells_in_window[:, 0] -= up
                    cells_in_window[:, 1] -= left
                    model = ParticleModel(cells_in_window, window)
                    potential_measurements[k] = model.get_potential_score()
                    if cells_in_window.size != 0:
                        try:
                            diagram = VoronoiDiagram(cells_in_window, None)
                            hexagon_ratios[k] = diagram.number_of_good_cells / diagram.number_of_regions
                        except (RuntimeError, ValueError):
                            hexagon_ratios[k] = 0

            window_positions[k, :] = np.array([up + bottom, left + right]) // 2

    cell_densities = np.divide(
        np.asarray(cell_numbers, dtype=np.float32), bright_areas,
        out=np.zeros_like(cell_numbers, dtype=np.float32), where=(bright_areas != 0))
    um_per_pixel = UM_PER_PIXEL
    cell_densities /= (um_per_pixel / 1000) ** 2

    if np.any(np.isnan(cell_densities)):
        plt.figure()
        plt.imshow(bitmap)
        plt.show()

        logging.info(cell_numbers)
        logging.info(bright_areas)
        logging.info(um_per_degree, um_per_pixel)
        raise RuntimeError

    return window_positions, cell_numbers, cell_densities, bright_area_ratios, hexagon_ratios, potential_measurements


def process_single_subject(
        montage_mosaic,
        input_path: Path = None,
        is_restarted: bool = False,
        output_path: Path = None,
        montage_path: Path = None,
        condition_on_images: Callable[[ImageFile], bool] = None,
        subject_id: int = None
) -> None:
    """
    Process images from a single subject, split them into regions of interest, and detect cells.

    This function takes all the images from the input directory, splits them into regions of interest (ROIs),
    and detects cells in each ROI. It saves the ROI with labeled coordinates in the specified output directory
    along with global statistics on this subject, such as photoreceptor density distribution plots.

    :param montage_mosaic: The montage mosaic to use for processing.
    :type montage_mosaic: Any
    :param input_path: The input directory containing images, defaults to None.
    :type input_path: Path, optional
    :param is_restarted: Indicates if the processing is restarted due to previous errors, defaults to False.
    :type is_restarted: bool, optional
    :param output_path: The output directory to save processed images, defaults to None.
    :type output_path: Path, optional
    :param montage_path: The directory containing montage images, defaults to None.
    :type montage_path: Path, optional
    :param condition_on_images: A callable that takes an ImageFile and returns True if the image should be processed, defaults to None.
    :type condition_on_images: Callable[[ImageFile], bool], optional
    :param subject_id: The ID of the subject being processed, defaults to None.
    :type subject_id: int, optional
    """
    input_dir = str(input_path)
    montage_dir = str(montage_path)
    output_dir = str(output_path)
    global_threshold_container = GlobalThresholdContainer()
    global_threshold_container.reset()

    logging.info('Obtaining axial lengths...')
    try:
        data = pd.ExcelFile(AXIAL_LENGTH_FILE)
        dfs = {sheet_name: data.parse(sheet_name) for sheet_name in data.sheet_names}
        table = dfs['Healthy']
        aoslo_ids = table['AOSLO ID']
        axial_lengths = table['AL D (mm)']
    except FileNotFoundError:
        warnings.warn(f'axial lengths not found at {AXIAL_LENGTH_FILE}')
        axial_lengths = None
    logging.info('Done.')

    if subject_id is None:
        startpos = input_dir.find('Subject') + len('Subject')
        endpos = input_dir.find('_Session')
        try:
            subject_id = int(input_dir[startpos: endpos])
        except ValueError:
            startpos = input_dir.find('Subject') + len('Subject')
            endpos = input_dir.find('\\Session')
            try:
                subject_id = int(input_dir[startpos: endpos])
            except ValueError:
                pass

    model_axial_length = 24
    if subject_id is not None and axial_lengths is not None:
        try:
            actual_axial_length = axial_lengths[aoslo_ids == subject_id].iloc[0]
            if pd.isna(actual_axial_length):
                actual_axial_length = model_axial_length
        except IndexError:
            actual_axial_length = model_axial_length
    else:
        actual_axial_length = model_axial_length

    um_per_degree = UM_PER_DEGREE * actual_axial_length / model_axial_length

    try:
        # it turned out that the true um per degree ratio is 291, not 300
        # it also depends on the ratio of true axial length to the model value of 24 mm
        # def condition_on_images(datafile: ImageFile) -> bool:
        #     "montaged" in str(datafile)

        # For montage instead of condition on image, we could have the exact position
        # of the center of the big image and the relative image' locations
        # new_way, old_way, single
        montage = "new_way"
        # split all the images onto regions of interest
        if montage == "new_way":
            def no_montage_condition(datafile: ImageFile) -> bool:
                return not "montaged" in str(datafile)
            image_files = select_images(input_dir, condition=no_montage_condition)
            # if montage_dir:
            #     montage_path = os.path.join(input_dir, montage_dir)
            # else:
            #     montage_path = os.path.join(input_dir, "montaged_corrected")
            # with open(os.path.join(montage_path, "shifts.pickle"), 'rb') as handle:
            #     shifts = pickle.load(handle)
            # with open(os.path.join(montage_path, "center.pickle"), 'rb') as handle:
            #     center = pickle.load(handle)
            # confocal_image = os.path.join(montage_path, MONTAGE_IMAGE)
            # shape = cv2.imread(confocal_image, cv2.IMREAD_GRAYSCALE).shape
            roi_infos_confocal, roi_infos_cs = get_roi_from_shift_images(input_dir, image_files, \
                montage_mosaic)
        elif montage == "old_way":
            image_files = select_images(input_dir)
            roi_infos_confocal, roi_infos_cs = get_roi_from_montaged_images(input_dir, image_files)
            image_files[0].read_data(input_dir)
            shape = image_files[0].data.shape
            total_image = np.zeros((shape[0], shape[1], 3))
        elif montage == "single":
            def default_condition(datafile: ImageFile) -> bool:
                return datafile.squared_eccentricity <= OUTER_FOVEA_BORDER ** 2
            image_files = select_images(input_dir, condition=default_condition)
            roi_infos_confocal, roi_infos_cs = get_roi_from_images(input_dir, image_files)

        # save the original ROI for further manual labelling, if needed
        # save also the dark region masks
        for i, roi_info in enumerate(roi_infos_confocal + roi_infos_cs):
            roi_imagefile = roi_info.get_image_file(image_files, i)
            # print(i, roi_imagefile)
            roi_imagefile.postfix += '_original'
            roi_imagefile.set_data(roi_info.original_image)
            roi_imagefile.write_data(output_dir)

            if roi_imagefile.modality == ImageModalities.CO.value:
                roi_imagefile.postfix = roi_imagefile.postfix.replace('_original', '_bitmap', 1)
                roi_imagefile.set_data(roi_info.bitmap * 255)
                roi_imagefile.write_data(output_dir)

            if roi_imagefile.modality == ImageModalities.CO.value:
                roi_imagefile.postfix = roi_imagefile.postfix.replace('_bitmap', '_preprocessed', 1)
                roi_imagefile.set_data(roi_info.preprocessed_image)
                roi_imagefile.write_data(output_dir)

            # plt.figure()
            # plt.imshow(roi_imagefile.data)
            # plt.show()

        # Note that if the size of the images is 1.5 degrees, than the edge ROI of the neighboring images are considered
        # as overlapping, and the density in these cases
        roi_center_coordinates = np.array([roi_info.center_coordinates_in_degrees for roi_info in roi_infos_confocal],
                                          dtype=DEGREE_COORDINATES_DATATYPE)

        roi_coordinate_in_pixels = convert_global_coordinates_deg2pixel(roi_center_coordinates,
                                                                        roi_infos_confocal[0].shape,
                                                                        PIXELS_PER_DEGREE)
        # determine size of the global bitmap in pixels
        vertical_size = np.max(roi_coordinate_in_pixels[:, 0]) + roi_infos_confocal[0].shape[0]
        horizontal_size = np.max(roi_coordinate_in_pixels[:, 1]) + roi_infos_confocal[0].shape[0]
        global_bitmap = np.zeros((vertical_size, horizontal_size), dtype=bool)

        # reset the thresholds for each new subject
        global_thresholds = GlobalThresholdContainer()
        global_thresholds.reset()

        window_x_coordinates = []
        window_y_coordinates = []
        window_cell_numbers = []
        window_cell_densities = []
        window_bright_area_ratios = []
        window_hexagon_ratios = []
        window_average_potentials = []
        total_number_of_cells = 0
        logging.info('Detecting cones in ROI')
        tr = trange(len(roi_infos_confocal), desc='Detecting cones in ROI')
        for i in tr:
            # eccentricity must be smaller than 3
            eccentricity = np.sqrt(np.sum(roi_infos_confocal[i].center_coordinates_in_degrees ** 2))
            if eccentricity <= OUTER_FOVEA_BORDER:
                retina_bitmap = roi_infos_confocal[i].bitmap
                # if np.count_nonzero(retina_bitmap) < retina_bitmap.size // 3:
                #     print('Too little bright space on the image, discard it.')
                #     continue

                # plt.figure()
                # plt.imshow(roi_infos_confocal[i].preprocessed_image, cmap='gray')
                # plt.show()

                is_close_to_fovea = np.sum(roi_infos_confocal[i].center_coordinates_in_degrees ** 2) < OUTER_FOVEA_BORDER ** 2
                reliable_coordinates, questionable_coordinates = \
                    get_cell_coordinates_on_image(roi_infos_confocal[i].preprocessed_image, is_close_to_fovea)
                total_number_of_cells += reliable_coordinates.shape[0]

                window_coordinates, window_counts, window_densities, window_areas, hexagon_ratios, potential_qualities = \
                    get_densities_across_roi(reliable_coordinates, roi_infos_confocal[i].preprocessed_image,
                                             roi_infos_confocal[i].bitmap, um_per_degree)

                # print(window_coordinates,
                #       roi_infos_confocal[i].center_coordinates_in_degrees,
                #       roi_infos_confocal[i].shape,
                #       roi_infos_confocal[i].size_in_degrees)
                window_coordinates_in_degrees = pixel2deg(window_coordinates,
                                                          roi_infos_confocal[i].center_coordinates_in_degrees,
                                                          roi_infos_confocal[i].shape,
                                                          roi_infos_confocal[i].size_in_degrees)
                # print(image_files[0].data.shape, roi_infos_confocal[i].shape, roi_infos_confocal[i].coordinates_on_image, roi_infos_confocal[i].size_in_degrees)
                # print(window_coordinates)
                # print(window_coordinates_in_degrees)
                # print()

                with np.printoptions(precision=1):
                    tr.set_postfix({'shift': f'{roi_infos_confocal[i].center_coordinates_in_degrees}',
                                    'number of cells': reliable_coordinates.shape[0]})

                roi_infos_confocal[i].set_cell_count(reliable_coordinates.shape[0])

                if not is_detected_well(reliable_coordinates, np.count_nonzero(retina_bitmap)):
                    warnings.warn(f'Bad detection at coordinates {roi_infos_confocal[i].center_coordinates_in_degrees}',
                                  RuntimeWarning)
                    # plt.figure()
                    # plt.imshow(roi_infos_confocal[i].original_image, cmap='gray')
                    # plt.title(roi_infos_confocal[i].center_coordinates_in_degrees)
                    # plt.show()
                    continue

                # mark this area on the global image
                try:
                    global_bitmap[
                    roi_coordinate_in_pixels[i, 0]: roi_coordinate_in_pixels[i, 0] + roi_infos_confocal[i].shape[0],
                    roi_coordinate_in_pixels[i, 1]: roi_coordinate_in_pixels[i, 1] + roi_infos_confocal[i].shape[1]
                    ] = retina_bitmap
                except ValueError:
                    logging.info(roi_infos_confocal[i].center_coordinates_in_degrees)
                    logging.info(roi_coordinate_in_pixels[i, 0],
                          roi_infos_confocal[i].shape[0],
                          roi_coordinate_in_pixels[i, 0] + roi_infos_confocal[i].shape[0])
                    logging.info(roi_coordinate_in_pixels[i, 1],
                          roi_infos_confocal[i].shape[1],
                          roi_coordinate_in_pixels[i, 1] + roi_infos_confocal[i].shape[1])
                    logging.info(retina_bitmap)
                    raise

                # mark photoreceptor locations as green squares
                labeled_image = highlight_coordinates_on_image(roi_infos_confocal[i].original_image,
                                                               reliable_coordinates,
                                                               marker_color=np.array([0., 1., 0.]),
                                                               marker_size=2)
                # # mark hidden photoreceptor locations as red squares
                # labeled_image = highlight_coordinates_on_image(labeled_image,
                #                                                questionable_coordinates,
                #                                                marker_color=np.array([1., 0., 0.]),
                #                                                marker_size=1)

                # save the results as a labeled image and a csv file with coordinates
                roi_imagefile = roi_infos_confocal[i].get_image_file(image_files, i)
                roi_imagefile.postfix += '_labeled'
                roi_imagefile.set_data(labeled_image)
                roi_imagefile.write_data(output_dir)

                if montage == "old_way":
                    x = roi_imagefile.crop_x_position
                    y = roi_imagefile.crop_y_position
                    total_image[y:y+185, x:x+185, :] = roi_imagefile.data

                roi_coordinatefile = CoordinatesFile(datafile=roi_imagefile)
                roi_coordinatefile.extension = 'csv'
                roi_coordinatefile.set_data(reliable_coordinates)
                roi_coordinatefile.write_data(output_dir)

                window_x_coordinates += list(window_coordinates_in_degrees[:, 1])
                window_y_coordinates += list(window_coordinates_in_degrees[:, 0])
                window_cell_numbers += list(window_counts)
                window_cell_densities += list(window_densities)
                window_bright_area_ratios += list(window_areas)
                window_hexagon_ratios += list(hexagon_ratios)
                window_average_potentials += list(potential_qualities)

        logging.info(f'Found {total_number_of_cells} cells.')

        logging.info('Displaying and saving the resulting figures...')
        cv2.imwrite(os.path.join(output_dir, RETINA_AREA_PATTERN), np.flipud(np.rot90(global_bitmap)) * 255)

        if montage == "old_way":
            cv2.imwrite(os.path.join(output_dir, "montaged_labeled.png"), total_image)
        with open(os.path.join(output_dir, 'counts.txt'), 'w') as f:
            for x, y, count, density in zip(window_x_coordinates, window_y_coordinates, window_cell_numbers,
                                            window_cell_densities):
                f.write(f'{x :0.3f} {y :0.3f} {count} {density}\n')
        
        # instead of computing the number of cells in each ROI, we plot the histogram of their global density
        resolution = (AVERAGING_SIZE_IN_UM + 5) / um_per_degree

        plot_dir = os.path.join(input_dir, Parser.configs.get("CellDetection", "__output_dir_atms"))
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        possible_center_positions, potential_criteria = plot_3d_distribution(window_x_coordinates, window_y_coordinates,
                                                                             window_cell_densities, resolution,
                                                                             plot_dir, save_images=True,
                                                                             show_images=True,
                                                                             bright_area_ratios=window_bright_area_ratios,
                                                                             hexagon_cell_ratios=window_hexagon_ratios,
                                                                             average_potentials=window_average_potentials)

        for file in os.listdir(input_dir):
            if file.startswith('Foveal'):
                os.remove(os.path.join(input_dir, file))

        if possible_center_positions is not None:
            for center_pos, potential in zip(possible_center_positions, potential_criteria):
                min_dist = 1e6
                min_image = None
                for roi in roi_infos_confocal:
                    dist = np.sum((roi.center_coordinates_in_degrees - center_pos) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        min_image = roi.original_image

                plt.figure()
                plt.imshow(min_image, cmap='gray')
                title = f'Foveal image at position ({center_pos[1] :0.3f}, {center_pos[0] :0.3f}), {potential = }'
                plt.title(title)
                # James: I don't know why those plots are made so avoid to save them
                # plt.savefig(os.path.join(input_dir, title + '.png'))
                plt.close()
                # plt.show()

        logging.info('done.')

    except ModuleNotFoundError:
        if is_restarted:
            raise
        else:
            # remove all the numba cache that for some reason prevents from loading some modules
            logging.info('An error occurred, remove all cache and restart. After that procedure the detection for the first '
                  'image will take much more time than for the other ones, it is normal.')

            cv_engine_folder = os.path.abspath(os.path.join(__file__, '..'))
            for entry in os.scandir(cv_engine_folder):
                if entry.is_dir():
                    if entry.name == '__pycache__':
                        shutil.rmtree(entry.path)
                    inner_cache_folder = os.path.join(entry.path, '__pycache__')
                    if os.path.exists(inner_cache_folder):
                        shutil.rmtree(inner_cache_folder)

            process_single_subject(montage_mosaic, input_path=input_dir, is_restarted=True, output_path=output_dir)

# def get_closest_center_images(all_roi_infos_confocal, all_roi_infos_cs, shifts, center):
#     import pdb
#     pdb.set_trace()

#     for roi_info_confocal in all_roi_infos_confocal:
#         center_coordinates_in_pixels = roi_info_confocal.center_coordinates_in_degrees * PIXELS_PER_DEGREE
#         center_difference_in_pixels =
#         center_coordinates_in_pixels = center_coordinates_in_pixels


#     return roi_infos_confocal, roi_infos_cs

def process_many_subjects(
        common_input_dir: str,
        condition: Callable[[str], bool],
        common_output_dir: str = None,
        condition_on_images: Callable[[ImageFile], bool] = None
) -> None:
    """
    Process images from multiple subjects based on specified conditions.

    This function iterates over directories in the common input directory, applies a condition to each directory,
    and processes the images within those directories if the condition is met. Optionally, it can save the processed
    images to a common output directory.

    :param common_input_dir: The directory containing subject directories.
    :type common_input_dir: str
    :param condition: A callable that takes a subject directory name and returns True if the directory should be processed.
    :type condition: Callable[[str], bool]
    :param common_output_dir: The directory to save processed images, defaults to None.
    :type common_output_dir: str, optional
    :param condition_on_images: A callable that takes an ImageFile and returns True if the image should be processed, defaults to None.
    :type condition_on_images: Callable[[ImageFile], bool], optional
    """
    for subject_dir in os.listdir(common_input_dir):
        if condition(subject_dir):
            logging.info(f'Processing images from {subject_dir}...')

            if common_output_dir is not None:
                output_dir = os.path.join(common_output_dir, subject_dir)
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
            else:
                output_dir = None

            process_single_subject(input_dir=os.path.join(common_input_dir, subject_dir),
                                   output_dir=output_dir,
                                   condition_on_images=condition_on_images)