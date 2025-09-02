from typing import List
from pathlib import Path
import numpy as np
import re

from src.shared.datafile.coordinates_file import CoordinatesFile

MDRNN_CONE_DETECTOR_OUT_DIR = 'output_cone_detector'

def get_cone_csv_file_path(output_path: Path) -> Path:
    """
    Get the path where the csv files from the MDRNN cone detection is

    :param output_path: output path of the cone detection
    :type output_path: str
    :return: the path where the csv files are
    :rtype: str
    """

    cone_dir = MDRNN_CONE_DETECTOR_OUT_DIR
    return output_path / cone_dir / "algorithmMarkers"

def read_cones_from_csv(csv_file: CoordinatesFile, file_path: Path | str) -> List[List[int]]:
    """
    Get the number of cones detected for each patch

    :param file: the file as a csv where the cone center are
    :type file: CoordinatesFile
    :param path: the path where the csv file is
    :type path: Path | str
    :return: an array with the cone locations as (y,x) for OpenCV images
    :rtype: List[List[int]]
    """
    csv_file.read_data(file_path)
    return csv_file.data.tolist()

def get_best_cone_data(
    file: CoordinatesFile,
    primary_path: Path,
    secondary_path: Path | None = None,
    third_path: Path | None = None) -> List[List[int]]:
    """
    Get the best cone data from multiple paths.

    This method reads cone data from the primary path and optionally from secondary and third paths.
    It returns the data with the highest number of cones detected.

    :param file: The coordinates file to read cone data from.
    :type file: CoordinatesFile
    :param primary_path: The primary path to read cone data from.
    :type primary_path: Path
    :param secondary_path: The secondary path to read cone data from (optional).
    :type secondary_path: Path, optional
    :param third_path: The third path to read cone data from (optional).
    :type third_path: Path, optional
    :return: The cone data with the highest number of cones detected.
    :rtype: List[List[int]]
    """
    best_cones_data = read_cones_from_csv(file, primary_path)
    best_number = len(best_cones_data)

    if secondary_path:
        second_cones_data = read_cones_from_csv(file, get_cone_csv_file_path(secondary_path))
        try:
            number_second_cones = len(second_cones_data)
        except TypeError:
            number_second_cones = 0
        if number_second_cones > best_number:
            best_cones_data = second_cones_data
            best_number = number_second_cones

    if third_path:
        third_cones_data = read_cones_from_csv(file, get_cone_csv_file_path(third_path))
        try:
            number_third_cones = len(third_cones_data)
        except TypeError:
            number_third_cones = 0
        if number_third_cones > best_number:
            best_cones_data = third_cones_data

    return best_cones_data
