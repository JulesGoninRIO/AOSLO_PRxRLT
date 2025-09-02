import os
import re
from pathlib import Path


def correcting_pattern(src: str,
                       filename: str,
                       wrong_pattern: str,
                       correct_pattern: str) -> None:
    """
    Correct the pattern of a string filename

    :param src: the directory where the file is
    :type src: str
    :param filename: the name of the file to correct
    :type filename: str
    :param wrong_pattern: the wrong pattern to correct
    :type wrong_pattern: str
    :param correct_pattern: the pattern to correct to
    :type correct_pattern: str
    """

    new_filename = re.sub(wrong_pattern, correct_pattern, filename)
    os.rename(os.path.join(src, filename), os.path.join(src, new_filename))


def correct_modality_names(src: Path) -> None:
    """
    Corrects the files with the correct modality names for
    850nm -> OA850nm and 790nm -> OA790nm

    :param src: the dir to look at
    :type src: the epath where to look for the files
    """

    for filename in os.listdir(src):
        wrong_pattern = "_850nm"
        correct_pattern = "_OA850nm"
        if wrong_pattern in filename:
            correcting_pattern(src, filename, wrong_pattern, correct_pattern)

        wrong_pattern = "_790nm"
        correct_pattern = "_OA790nm"
        if wrong_pattern in filename:
            correcting_pattern(src, filename, wrong_pattern, correct_pattern)
