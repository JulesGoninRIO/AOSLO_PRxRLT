from src.shared.datafile.datafile_constants import ImageModalities
from typing import List, Union, Dict
import os
from src.shared.datafile.datafile import DataFile
from pathlib import Path
import logging
import copy

def safe_file_type_constructor(filename: str, constructor: callable):
    """
    Safely construct a file type object.

    This function attempts to construct a file type object using the provided constructor
    and filename. If a ValueError occurs during construction, it returns None.

    :param filename: The name of the file to construct the object from.
    :type filename: str
    :param constructor: The constructor function to create the file type object.
    :type constructor: Callable[[str], Any]
    :return: The constructed file type object or None if a ValueError occurs.
    :rtype: Any or None
    """
    try:
        return constructor(filename)
    except ValueError:
        return None

def find_files(
    input_dir: Path,
    file_type: DataFile,
    modalities: Union[ImageModalities, List[ImageModalities]] = None) -> List[DataFile]:
    """
    Find the files from a directory while ignoring subdirectories

    :param input_dir: the folder where the files are
    :type input_dir: str
    :param extension: condtion on extension of the files to find
    :type extension: str, optional
    :param modalities: the modalities of the files we want to find
    :type modalities: List[ImageModalities], optional
    :return: the files from the folder as a list
    :rtype: List[str]
    """

    resulting_files = []
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if modalities:
        if not isinstance(modalities, list):
            modalities = [modalities]
        resulting_files = []
        for modality in modalities:
            resulting_files.extend([f for f in files if modality.value in f])
    else:
        resulting_files = files
    return [f for f in (safe_file_type_constructor(f, file_type) for f in resulting_files) if f is not None]

def find_corresponding_files(
    input_dir: Path,
    modality_files: List[DataFile],
    modalities: Union[ImageModalities, List[ImageModalities]]) -> Dict[DataFile, List[DataFile]]:
    """
    Finds the other modality images that corresponding to a modality images

    :param input_dir: the folder where the images are
    :type input_dir: str
    :param modality_images: the images from a certain modality
    :type modality_images: List[str]
    :param modalities: the other modalities to find images
    :type modalities: List[ImageModalities]
    :return: a dictionnary with the modality images as key and the other
    modalities found as a list of their names
    :rtype: Dict[str, List[str]]
    """

    corresponding_images = {}
    directory_files = set(os.listdir(input_dir))
    for modality_file in modality_files:
        # look for the other corresponding modality images and stores them
        correspond_modality_images = []
        if not isinstance(modalities, list):
            modalities = [modalities]
        for modality in modalities:
            # change the modality of the file
            modality_file_copy = copy.copy(modality_file)
            modality_file_copy.modality = modality.value
            if str(modality_file_copy) in directory_files:
                correspond_modality_images.append(modality_file_copy)
            else:
                # the other modality image do not exist
                logging.error(f"{str(modality_file)} image has not been found, \
                please make sure the folder contains every \
                images or remove {str(modality_file)} from the folder.")

        corresponding_images[modality_file] = correspond_modality_images

    return corresponding_images
