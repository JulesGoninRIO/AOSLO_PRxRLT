import numpy as np
from typing import Tuple, List
import os
import pickle
import logging
from pathlib import Path
import re
import pandas as pd

from src.shared.helpers.direction import Direction
from src.cell.processing_path_manager import ProcessingPathManager
from src.cell.analysis.constants import PIXELS_PER_DEGREE, RESULT_NAME
from src.configs.parser import Parser
from src.shared.computer_vision.point import Point

def eccentricity(point: np.ndarray, direction: Direction | None, center: Point, right_eye: bool = True) -> float:
    """
    Compute the eccentricity of a point with respect to the center of the mosaic.
    Distance in degrees, with sign indicating which side of the center the point is on, in the specified direction (if no `direction` provided, sign is always positive). 

    If `right_eye` is True, the sign is reversed for the X axis, to ensure positive x eccentricity always corresponds to nasal and negative x eccentricity to temporal (recall: as AOSLO images are mirrored, nasal is right meridian for left eyes, and left meridian for right eyes, hence the sign flip; check `.../montaged_corrected/mosaic_fundus*.tif` and recall that AOSLO montage is the non _flipped one to convince yourself). For the Y axis, the sign is always negative for superior and positive for inferior.

    :param point: The point for which to compute the eccentricity.
    :type point: np.ndarray
    :param direction: The direction in which to compute the eccentricity.
    :type direction: Direction | None
    :param center: The center of the mosaic.
    :type center: Point
    :param right_eye: Whether the right eye is being processed, defaults to True.
    :type right_eye: bool
    """
    diff_x = point[0] - center.x
    diff_y = point[1] - center.y
    if direction is not None:
        if direction.is_X:
            sign = 1 if diff_x >= 0 else -1
            if right_eye:
                sign *= -1
        elif direction.is_Y:
            sign = 1 if diff_y >= 0 else -1
    else:
        sign = 1
    return sign * np.sqrt(diff_x**2 + diff_y**2) / PIXELS_PER_DEGREE

def sort_by_ecc(densities: np.ndarray | List) -> np.ndarray:
    """
    Sort densities by eccentricity, assuming that the first column is the eccentricity and the second is the density.
    """
    return np.array(sorted(densities, key=lambda x: x[0]))

def cut_array_borders(array: np.array, nan_values: np.array) -> np.array:
    """
    Cut the borders of an array based on NaN values.

    This function trims the borders of the given array where NaN values are present at the start or end.

    :param array: The input array to be trimmed.
    :type array: np.array
    :param nan_values: An array indicating the positions of NaN values.
    :type nan_values: np.array
    :return: The trimmed array.
    :rtype: np.array
    """
    if np.any(nan_values == len(array)-1):
        cut = 1 + np.sum(nan_values[:-1] == np.arange(len(array)-2, len(array)-1-len(nan_values), -1))
        return array[:-cut]
    elif np.any(nan_values == 0):
        cut = 1 + np.sum(nan_values[1:] == np.arange(1, 1+len(nan_values)))
        return array[cut:]
    return array

def interpolate_nans(array: np.array) -> np.array:
    """
    Interpolate NaN values in an array.

    This function interpolates the NaN values in the given array using linear interpolation.

    :param array: The input array with NaN values to be interpolated.
    :type array: np.array
    :return: The array with NaN values interpolated.
    :rtype: np.array
    """
    ok = ~np.isnan(array)
    xp = ok.ravel().nonzero()[0]
    fp = array[~np.isnan(array)]
    x = np.isnan(array).ravel().nonzero()[0]
    array[np.isnan(array)] = np.interp(x, xp, fp)
    return array


def replace_nans(array: np.array, second_array: np.array = None) -> Tuple[np.array, np.array]:
    """
    Replace nan values in an array by either:
    If the nan values are on border, cut the array
    If in the middle, replace by mean of neighbor values

    if a second array is given (for spearman computations), if we cut nan values
    in the first or second array, it will also cut in the other array at same position
    so that we keep same array length

    :param array: the first array that might contain nan values
    :type array: np.ndarray
    :param second_array: the second array that might contain nan values, defaults to None
    :type second_array: np.ndarray, optional
    :return: the updated arrays without any nan values left
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if second_array is not None and not isinstance(second_array, np.ndarray):
        second_array = np.array(second_array)
  



    return remove_paired_nans(array1 = array, array2 = second_array) if second_array is not None else array



def remove_paired_nans(array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove all positions where either array has NaN values.
    This is the most statistically sound approach for correlation analysis.
    
    :param array1: First array
    :param array2: Second array
    :return: Both arrays with NaN positions removed
    """
    # Create mask for valid (non-NaN) positions in both arrays
    valid_mask = ~(np.isnan(array1) | np.isnan(array2))
    
    return array1[valid_mask], array2[valid_mask]

def load_checkpoint(checkpoint_path: str, *pickle_objects) -> List:
    """
    Load pickle objects from a checkpoint directory for long analysis

    :param checkpoint_path: the path where to load the pickle objects
    :type checkpoint_path: str
    :params pickle_objects: the objects to load as pickles
    :return: the list of the loaded pickle objects
    :rtype: List
    """
    objects = []
    for pickle_object in pickle_objects:
        try:
            pickle_object_name = os.path.join(
                checkpoint_path, f"{str(pickle_object)}.pickle")
            with open(pickle_object_name, 'rb') as handle:
                pickle_object = pickle.load(handle)
            objects.append(pickle_object)
        except FileNotFoundError:
            logging.error(f"The file {pickle_object_name} cannot be found, \
                            make sure it exists.")

    return objects

def gather_results_deprecated(path, dir_to_process = None):
    """
    Gathers results from all session directories and returns the final DataFrame.

    This method iterates through all subject and session directories, processes the session data,
    and concatenates the results into a single DataFrame.

    :return: The final DataFrame containing all results.
    :rtype: pd.DataFrame
    """
    result_list = []
    resulting_df = pd.DataFrame()
    subject_dirs = _find_subject_dirs(path)
    for subject_dir in subject_dirs:
        subject_path = os.path.join(path, subject_dir)
        try:
            session_dirs = [file for file in os.listdir(subject_path) if "Session"
                            in file and os.path.isdir(os.path.join(subject_path, file))]
        except FileNotFoundError:
            print("oe")
        for session_dir in session_dirs:
            session_path = os.path.join(subject_path, session_dir)
            if dir_to_process and os.path.normpath(session_path) == os.path.normpath(dir_to_process):
                continue
            result = _process_session_dir(Path(session_path))
            if result is not None:
                subject = re.search(r"Subject(\d+)", subject_dir).group(0)
                session = re.search(r"Session(\d+)", session_dir).group(0)
                result_list.append(subject + "_" + session)
                resulting_df = pd.concat([resulting_df, result]) if not resulting_df.empty else result

    return resulting_df

def _find_subject_dirs(path: Path):
    """
    Finds and returns subject directories.

    This method searches for directories within the parent path that contain the word "Subject".
    If no such directories are found, it adjusts the parent path and searches again.

    :return: A list of subject directories.
    :rtype: list
    """
    subject_dirs = [file for file in os.listdir(path) if "Subject"
                    in file and os.path.isdir(os.path.join(path, file))]
    if len(subject_dirs) == 0:
        if "session" in path.name.lower():
            if "subject" in path.name.lower():
                path = Path(path).parent
            else:
                path = Path(path).parent.parent
        subject_dirs = [file for file in os.listdir(path) if "Subject"
                        in file and os.path.isdir(os.path.join(path, file))]
    return subject_dirs

def _process_session_dir(session_path):
    """
    Processes a session directory and returns the result DataFrame.

    This method reads the results from a session directory, filters the data, and returns it as a DataFrame.

    :param session_path: The path to the session directory.
    :type session_path: Path
    :return: The result DataFrame or None if the file is not found or empty.
    :rtype: pd.DataFrame or None
    """
    path_manager = ProcessingPathManager(session_path)
    result_file = path_manager.density.path / "results.csv"
    if not os.path.isfile(result_file):
        return None
    try:
        result = pd.read_csv(result_file, skiprows=1)
    except pd.errors.EmptyDataError:
        return None
    if result.empty:
        return None
    return result[abs(result.location) <= 10]

def gather_results(path: Path, sess_dir_to_process: Path | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gathers results from all session directories and returns the final DataFrame.

    This method iterates through all subject and session directories, processes the session data,
    and concatenates the results into a single DataFrame.

    :return: The final DataFrame containing all results as well as list of considered Subject/Session.
    :rtype: Tuple[pd.DataFrame, List[str]]
    """

    result_list = []
    resulting_df = pd.DataFrame()
    sess_dir_to_process = Path(sess_dir_to_process).resolve() if sess_dir_to_process else None

    for result_file in path.glob(f'Subject*/Session*/{Parser.get_density_analysis_dir()}/{RESULT_NAME}'):
        session_dir = result_file.parent.parent
        if sess_dir_to_process is not None and session_dir.resolve() == sess_dir_to_process:
            continue
        try:
            result = pd.read_csv(result_file, skiprows=1)
        except pd.errors.EmptyDataError:
            continue
        if result.empty:
            continue

        subject = re.search(r'Subject(\d+)', session_dir.parent.name).group(0)
        session = re.search(r'Session(\d+)', session_dir.name).group(0)
        result_list.append(f'{subject}_{session}')

        result = result[abs(result['location']) <= 10]
        resulting_df = pd.concat([resulting_df, result]) if not resulting_df.empty else result

    return resulting_df, result_list