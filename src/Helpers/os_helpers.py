import glob
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

from src.Helpers.constants import PREVIOUS_RUN_DIR
from src.PostProc_Pipe.Helpers.datafile_classes import (ImageFile,
                                                        ImageModalities)


def deplace_files(source_dir: str, destination_dir: str, files: List[str]) -> None:
    """
    Deplace an image from the source directory to the destination one. If the
    destination folder does not exist, it will create one.

    :param source_dir: source of the file directory
    :type source_dir: str
    :param destination_dir: destination of the file directory
    :type destination_dir: str
    :param files: files that are displaced
    :type files: List[str]
    """

    for f in files:
        if not os.path.isdir(destination_dir):
            os.makedirs(destination_dir, exist_ok=False)
        source = os.path.join(source_dir, f)
        destination = os.path.join(destination_dir, f)
        try:
            shutil.move(source, destination)
        except PermissionError:
            logging.warning(f"You are trying to remove a file: {source} which you don't "
                            "have access or you have it open somwhere. Please verify "
                            "the code or close the file.")


def deplace_everything(source_dir: str, destination_dir: str) -> None:
    """
    Deplace everything from the source directory to the destination one. If the
    destination folder does not exist, it will create one.

    :param source_dir: source of the file directory
    :type source_dir: str
    :param destination_dir: destination of the file directory
    :type destination_dir: str
    """

    files = [file for file in os.listdir(source_dir)]
    deplace_files(source_dir, destination_dir, files)


def copy_paste_files(source_dir: str, destination_dir: str, files: List[str]) -> None:
    """
    Copy Paste an image from the source directory to the destination one. If the
    destination folder does not exist, it will create one.

    :param source_dir: source of the file directory
    :type source_dir: str
    :param destination_dir: destination of the file directory
    :type destination_dir: str
    :param files: files that are displaced
    :type files: List[str]
    """

    for f in files:
        if not os.path.isdir(destination_dir):
            os.makedirs(destination_dir, exist_ok=False)
        source = os.path.join(source_dir, f)
        destination = os.path.join(destination_dir, f)
        try:
            shutil.copyfile(source, destination)
        except PermissionError:
            logging.warning(f"You are trying to remove a file: {source} which you don't "
                            "have access or you have it open somwhere. Please verify "
                            "the code or close the file.")
        except FileNotFoundError: continue


def get_folder_name(filename_dir: str, name: str) -> str:
    """
    Finds the name we want in the filename given. If we cannot find it, we
    search in the subfolders the files which has the name
    WARNING: the current implementation only supports lower case name

    :param filename_dir: the directory where we need to find the name in
    :type filename_dir: str
    :param name: their name to find (e.g. sesssion, subject)
    :type name: str
    :return: the name with number found (e.g. Session367, Subject1)
    :rtype: str
    """

    out_name = None
    try:
        # find the name
        out_name = re.search(
            name+r"(\d+)", filename_dir.lower()).group(0).capitalize()
    except AttributeError:
        # lok for files with the name in the subdirectory
        files = [f for f in os.listdir(filename_dir) if os.path.isfile(
            os.path.join(filename_dir, f))]
        for file_image in files:
            try:
                out_name = re.search(
                    name+r"(\d+)", file_image.lower()).group(0).capitalize()
                break
            except AttributeError:
                continue

    return out_name


def remove_file(src: Path) -> None:
    """
    Remove src file. Catch excpetions and write them as logging warning.

    :param src: file to delete
    :type src: Path
    """
    try:
        os.remove(src)
    except IsADirectoryError:
        logging.warning(f"You are trying to remove a directory: {src} and the "
                        "function expect to remove a file. Please verify the code, "
                        "the directory has not been removed")
    except FileNotFoundError:
        logging.warning(f"You are trying to remove a file: {src} which cannot be "
                        "found. Please verify the code.")
    except PermissionError:
        logging.warning(f"You are trying to remove a file: {src} which you don't "
                        "have access or you have it open somwhere. Please verify "
                        "the code or close the file.")


def remove_folder(src: Path) -> None:
    """
    Remove src folder and everything it contains. Catch excpetions and write them
    as logging warning.

    :param src: folder to delete
    :type src: Path
    """
    try:
        shutil.rmtree(src)
    except NotADirectoryError:
        logging.warning(f"You are trying to remove a file: {src} and the "
                        "function expect to remove a directory. Please verify the code, "
                        "the file has not been removed")
    except FileNotFoundError:
        logging.warning(f"You are trying to remove a directory: {src} which cannot be "
                        "found. Please verify the code.")
    except PermissionError:
        logging.warning(f"You are trying to remove a directory: {src} which you don't "
                        "have access or you have a file from it open somwhere. "
                        "Please verify the code.")


def remove_anything(src: Path) -> None:
    """
    Remove src (file or folder). Catch excpetions and write them as logging warning.

    :param src: file or folder to delete
    :type src: Path
    """
    try:
        os.remove(src)
    except FileNotFoundError:
        logging.warning(f"You are trying to remove a file or a  directory: {src} "
                        "which cannot be found. Please verify the code.")
    except PermissionError:
        try:
            shutil.rmtree(src)
        except PermissionError:
            logging.warning(f"You are trying to remove a file or a  directory: {src} "
                            "which you don't have access or you have a file from it "
                            "open somwhere. Please verify the code.")


def empty_folder(src=Path) -> None:
    """
    Remove all files or subdirectories inside a folder

    :param src: folder where files need to be deleted
    :type src: Path
    """
    for file in os.listdir(src):
        remove_anything(os.path.join(src, file))


def find_files(input_dir: str, extension: str = None,
               modalities: List[ImageModalities] = None) -> List[str]:
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
    files = [f for f in os.listdir(input_dir) if
             os.path.isfile(os.path.join(input_dir, f))]
    if extension:
        files = [f for f in files if f.endswith("." + extension)]
    if modalities:
        resulting_files = []
        for modality in modalities:
            resulting_files.extend([f for f in files if modality.value in f])
    else:
        resulting_files = files

    return resulting_files


def find_corresponding_images(input_dir: str, modality_images: List[str],
                              modalities: List[ImageModalities]) -> Dict[str, List[str]]:
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
    for modality_image_name in modality_images:
        # look for the other corresponding modality images and stores them
        correspond_modality_images = []
        modality_image = ImageFile(modality_image_name)
        for modality in modalities:
            # change the modality of the file
            modality_image.modality = modality.value
            if str(modality_image) in os.listdir(input_dir):
                correspond_modality_images.append(str(modality_image))
            else:
                # the other modality image do not exist
                logging.error(f"{str(modality_image)} image has not been found, \
                please make sure the folder contains every \
                images or remove {modality_image_name} from the folder.")

        corresponding_images[modality_image_name] = correspond_modality_images

    return corresponding_images


def start_step(process_dirs: List[str]) -> None:
    """
    Start each step by cleaning the process directory and displacing files in a
    new directory called

    :param process_dir: the directories to clean
    :type process_dir: List[str]
    """
    for process_dir in process_dirs:
        if os.path.exists(process_dir):
            previous_run_dir = PREVIOUS_RUN_DIR
            previous_run_path = os.path.join(
                Path(process_dir).parent, previous_run_dir)
            if not os.path.isdir():
                os.makedirs(previous_run_path, exists_ok=True)
            logging.info(f"Clearing output directory: {process_dir} \n Putting "
                         f"files in {previous_run_path}.")
            try:
                deplace_everything(process_dir, previous_run_path)
                logging.info(f"{process_dir} cleaned successfully")
            except (OSError, PermissionError):
                logging.error(f"Unable to remove files from the {process_dir} "
                              "directory. You may have a file or subdirectory open. "
                              "Please, close it, exit the folder and try again.")
                raise
        else:
            os.makedirs(process_dir, exists_ok=True)
