import glob
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

PREVIOUS_RUN_DIR = "previous_pipeline_run"
from src.shared.datafile.image_file import ImageFile
from src.shared.datafile.helpers import ImageModalities
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def has_in_name(string: str, name: str) -> bool:
    """
    Check whether the string given is in the name given

    :param string: the string to check
    :type string: str
    :param name: the name to check
    :type name: str
    :return: True if the string is in the name, else False
    :rtype: bool
    """
    return string in name


def test_single_subject(dir_to_process: str) -> str:
    """
    Test that a folder path given has one of the authorized given form and
    renames it

    :param dir_to_process: the path of the directory to process
    :type dir_to_process: str
    :return: the updated name
    :rtype: str
    """

    if isinstance(dir_to_process, Path):
        dir_to_process = str(dir_to_process)
    # first if we give the full path to only one subject

    if has_in_name("subject", os.path.basename(dir_to_process).lower()):
        if has_in_name("session", os.path.basename(dir_to_process).lower()):
            # handles subject...session... as input dir
            files = os.listdir(dir_to_process)
            dir_name = os.path.basename(dir_to_process).lower()
            try:
                session_name = re.search(
                    r"session(\d+)", dir_name).group(0).capitalize()
            except AttributeError:
                session_name = [re.search(r"Session(\d+)", file).group(0) for file
                                in os.listdir(dir_to_process) if re.search(r"Session(\d+)", file)
                                is not None][0].capitalize()
            try:
                subject_name = re.search(
                    r"subject(\d+)", dir_name).group(0).capitalize()
            except AttributeError:
                subject_name = [re.search(r"Subject(\d+)", file).group(0) for file
                                in os.listdir(dir_to_process) if re.search(r"Subject(\d+)", file)
                                is not None][0].capitalize()
            deplace_everything(dir_to_process, os.path.join(
                dir_to_process, session_name))
            new_name = os.path.join(
                Path(dir_to_process).parent, subject_name, session_name)
            try:
                os.rename(dir_to_process, new_name)
            except PermissionError:
                logging.error(
                    f"Trying to rename {dir_to_process} but you have "
                    "it open somehwere (File Explorer or terminal). "
                    "Please close it and restart")
                raise
            except FileNotFoundError:
                os.rename(dir_to_process, Path(new_name).parent)
            return new_name
        else:
            # handles subject... as input dir -> search for subdir
            subdirs = [
                os.path.join(dir_to_process, folder) for folder in
                os.listdir(dir_to_process) if "session" in folder.lower() and
                os.path.isdir(os.path.join(dir_to_process, folder))]
            dirs_to_process = []
            if len(subdirs) > 0:
                for subdir in subdirs:
                    dir_analyzed = test_single_subject(subdir)
                    if dir_analyzed is not None:
                        dirs_to_process.append(dir_analyzed)
            else:
                # no session dir inside of subjects -> create one and put everything in it
                try:
                    session_name = [re.search(r"Session(\d+)", file).group(0) for file
                                    in os.listdir(dir_to_process) if re.search(r"Session(\d+)", file)
                                    is not None][0].capitalize()
                except IndexError:
                    return None
                new_name = os.path.join(dir_to_process, session_name)
                deplace_everything(dir_to_process, new_name)
                dirs_to_process.append(new_name)
            return dirs_to_process

    elif has_in_name("session", os.path.basename(dir_to_process).lower()):
        # has subject../session..
        if has_in_name("subject", dir_to_process.lower()):
            try:
                # we process only one subject which has the correct path
                subject_number = re.search(
                    r"subject(\d+)", dir_to_process.lower()).group(0)
                try:
                    session_number = re.search(
                        r"session(\d+)", dir_to_process.lower()).group(0)
                    # handle: subject12/session45/
                    return dir_to_process
                except AttributeError:
                    # handle: subject12/session/
                    session_number = [
                        re.search(r"Session(\d+)", file).group(0)
                        for file in os.listdir(dir_to_process) if
                        re.search(r"Session(\d+)", file) is not None][0].capitalize()
                    new_name = os.path.join(
                        Path(dir_to_process).parent, session_number)
                    try:
                        os.rename(dir_to_process, new_name)
                    except PermissionError:
                        logging.error(
                            f"Trying to rename {dir_to_process} but you "
                            "have it open somehwere (File Explorer or terminal). "
                            "Please close it and restart")
                        raise
                    return new_name
            except AttributeError:
                try:
                    subject_name = [re.search(r"Subject(\d+)", file).group(0) for
                                    file in os.listdir(dir_to_process) if
                                    re.search(r"Subject(\d+)", file) is not None][0].capitalize()
                    try:
                        # this handle: "subject/session244/"
                        session_number = re.search(
                            r"session(\d+)", dir_to_process.lower()).group(0).capitalize()
                        new_name = os.path.join(Path(Path(dir_to_process).parent).parent,
                                                subject_name)
                        try:
                            os.rename(Path(dir_to_process).parent, new_name)
                        except PermissionError:
                            logging.error(
                                f"Trying to rename {dir_to_process} but you "
                                "have it open somehwere (File Explorer or terminal). "
                                "Please close it and restart")
                            raise
                        return new_name
                    except AttributeError:
                        try:
                            # handel subject/session/
                            session_base_name = os.path.basename(
                                dir_to_process)
                            subject_name = [re.search(r"Subject(\d+)", file).group(0)
                                            for file in os.listdir(dir_to_process) if
                                            re.search(r"Subject(\d+)", file) is not None][0].capitalize()
                            session_name = [re.search(r"Session(\d+)", file).group(0)
                                            for file in os.listdir(dir_to_process) if
                                            re.search(r"Session(\d+)", file) is not None][0].capitalize()
                            new_subject_name = os.path.join(Path(Path(dir_to_process).parent).parent,
                                                            subject_name)
                            try:
                                os.rename(
                                    Path(dir_to_process).parent, new_subject_name)
                            except PermissionError:
                                logging.error(
                                    f"Trying to rename {dir_to_process} but you "
                                    "have it open somehwere (File Explorer or terminal). "
                                    "Please close it and restart")
                                raise
                            new_session_name = os.path.join(new_subject_name,
                                                            session_name)
                            try:
                                os.rename(
                                    os.path.join(new_subject_name, session_base_name),
                                    new_session_name
                                )
                            except PermissionError:
                                logging.error(
                                    f"Trying to rename {new_subject_name} but you "
                                    "have it open somehwere (File Explorer or terminal). "
                                    "Please close it and restart")
                                raise
                            return new_session_name
                        except IndexError:
                            logging.error(
                                "Cannot find the subject number in files. "
                                "Please verify the data")
                            raise
                except IndexError:
                    logging.error(
                        "Cannot find the subject number in files. "
                        "Please verify the data")
                    raise
        else:
            try:
                # handles: session67/ and session/
                subject_name = [re.search(r"Subject(\d+)", file).group(0) for file
                                in os.listdir(dir_to_process) if re.search(r"Subject(\d+)", file)
                                is not None][0].capitalize()
                try:
                    session_name = re.search(
                        r"session(\d+)", dir_to_process).group(0).capitalize()
                except AttributeError:
                    session_name = [re.search(r"Session(\d+)", file).group(0) for file
                                    in os.listdir(dir_to_process) if re.search(r"Session(\d+)", file)
                                    is not None][0].capitalize()
                deplace_everything(dir_to_process, os.path.join(dir_to_process,
                                                                session_name))
                new_name = os.path.join(
                    Path(dir_to_process).parent, subject_name)
                out_name = os.path.join(new_name, session_name)
                try:
                    os.rename(dir_to_process, new_name)
                except PermissionError:
                    logging.error(
                        f"Trying to rename {dir_to_process} but you "
                        "have it open somehwere (File Explorer or terminal). "
                        "Please close it and restart")
                    raise
                return out_name
            except IndexError:
                logging.error(
                    "Cannot find the subject number in files. "
                    "Please verify the data")
                raise
    else:
        logging.error(
            f"The naming conventions is not respected for {dir_to_process}. "
            "Will ignore this folder.")
        return None


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
        except NotADirectoryError:
            # thumbs.db file
            pass


def empty_folder(src=Path) -> None:
    """
    Remove all files or subdirectories inside a folder

    :param src: folder where files need to be deleted
    :type src: Path
    """
    for file in os.listdir(src):
        remove_anything(os.path.join(src, file))


def find_files(
    input_dir: str, extension: str = None,
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
    files = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if extension:
        files = [f for f in files if f.endswith("." + extension)]
    if modalities:
        resulting_files = []
        for modality in modalities:
            resulting_files.extend([f for f in files if modality.value in f])
    else:
        resulting_files = files

    return resulting_files


def find_corresponding_images(
    input_dir: str, modality_images: List[str],
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
            logging.info(
                f"Clearing output directory: {process_dir} \n Putting "
                f"files in {previous_run_path}.")
            try:
                deplace_everything(process_dir, previous_run_path)
                logging.info(f"{process_dir} cleaned successfully")
            except (OSError, PermissionError):
                logging.error(
                    f"Unable to remove files from the {process_dir} "
                    "directory. You may have a file or subdirectory open. "
                    "Please, close it, exit the folder and try again.")
                raise
        else:
            os.makedirs(process_dir, exists_ok=True)

def verify_images(dirs_to_process: List[str]) -> None:
    """
    Verify that the images in the directories to process are from the same
    subjects and same session to avoid further errors

    :param dirs_to_process: list of directories that will be further use
    :type dirs_to_process: List[str]
    """

    for dir_to_process in dirs_to_process:
        verify_image_name_integrity(dir_to_process)

def verify_image_name_integrity(dir_to_process: str) -> None:
    """
    Verify that the images are having the proper subject and session names so
    that the patients are not mixed.

    :param dir_to_process: the directory where the files are
    ::type dirs_to_process: str
    """

    files = find_files(dir_to_process, "tif")
    subject_number = re.search(r"Subject(\d+)", dir_to_process).group(1)
    session_number = re.search(r"Session(\d+)", dir_to_process).group(1)
    for f in files:
        try:
            file_subject_number = re.search(r"Subject(\d+)", f).group(1)
            file_session_number = re.search(r"Session(\d+)", f).group(1)

            try:
                assert (file_subject_number == subject_number)
            except AssertionError:
                logging.error(
                    f"{f} is not from the same subject as "
                    f"the folder {subject_number}. Aborting")
                raise
            try:
                assert (file_session_number == session_number)
            except AssertionError:
                logging.error(
                    f"{f} is not from the same session as "
                    f"the folder {session_number}. Aborting")
                raise

        except AttributeError:
            logging.error(f"{f} is not of the format of ImageFile, Aborting")
            raise

def prepare_dataset(dir_to_process: str, blood_flow: bool) -> List[Path]:
    """
    Prepare the data for montaging by looking into the structure of the
    data and reorder the folder to be of the form:
    SubjectXXX -> SessionXXX -> images
    If blood_flow is selected, the search will be different, adapted to data
    from XT scans

    :param dir_to_process: the base directory where the files are
    :type dir_to_process: str
    :param blood_flow: whether we are doing the blood flow analysis
    :type blood_flow: bool
    :raises FileNotFoundError: if the given folder does not exists
    :return: all the session subfolders to analyze
    :rtype: List[str]
    """
    logging.info("prepare")
    if not os.path.isdir(dir_to_process):
        logging.error(
            "The folder given does not exists, please verify its name")
        raise FileNotFoundError

    logging.info(f"Processing data for {dir_to_process}")
    dirs_to_process = []

    if blood_flow:
        # no search for specific names for XT scans
        if 'subject' in os.path.basename(dir_to_process).lower() or 'session' in os.path.basename(dir_to_process).lower():
            return [dir_to_process]
        dirs_to_process = [
            folder for folder in os.listdir(dir_to_process) if
            os.path.isdir(os.path.join(dir_to_process, folder))
        ]
        return dirs_to_process

    # we have only one subject to process
    if has_in_name("subject", os.path.basename(dir_to_process).lower()) or \
            has_in_name("session", os.path.basename(dir_to_process).lower()):
        dir_to_process_analyzed = test_single_subject(dir_to_process)
        if dir_to_process_analyzed is not None:
            dirs_to_process.append(dir_to_process_analyzed)

    else:
        # we have a directory with multiple folders
        filenames = os.listdir(dir_to_process)
        for filename in filenames:
            dir_to_process_analyzed = test_single_subject(
                os.path.join(dir_to_process, filename))
            if dir_to_process_analyzed is not None:
                dirs_to_process.extend(dir_to_process_analyzed)

    verify_images(dirs_to_process)

    return [Path(dir_to_process) for dir_to_process in dirs_to_process]


def get_message(method: str,
                directory_name: str,
                already_done: List[str], #or path
                to_do: List[str], #or path
                checked: str) -> str:
    """
    Construct the messgae to ask the user for input based on which step of the pipeline and
    what has already been done.

    :param method: the step of the pipeline to aske the user for
    :type method: str
    :param directory_name: the name of the directory where the results of the pipeline might be
    :type directory_name: str
    :param already_done: the subects that have already this step done
    :type already_done: List[str]
    :param to_do: the subjects that are to do for this step
    :type to_do: List[str]
    :param checked: whether we wanted to do this pipeline step or not
    :type checked: str
    :return: the message given to the user
    :rtype: str
    """

    message_done = ""
    for file in already_done:
        if not isinstance(file, str):
            message_done += str(file) + " \n"
        else:
            message_done += file + " \n"
    message_to_do = ""
    for file in to_do:
        if not isinstance(file, str):
            message_to_do += str(file) + " \n"
        else:
            message_to_do += file + " \n"
    if checked == " not ":
        do = "do"
    else:
        do = "redo"
    message = (
        f"You have {checked} checked Do {method} but there is the {method} folder: "
        f"{directory_name} that does {checked} already exists for some folders to process. \n"
        f"{method} subject done: \n") + message_done + (f"{method} subject to do: \n") \
        + message_to_do + (f"Do you want to {do} {method} on those subjects?")
    return message

def check_step_done(dirs_to_process: List[str], step_dir: str) -> Tuple[List[str], List[str]]:
    """
    Check whether a step of the pipeline is done by looking into subdirectories
    whether there are the given files.

    :param dirs_to_process: the directories where to look for files
    :type dirs_to_process: List[str]
    :param step_dir: the specific subdirectory name of the step of the pipeline
    :type step_dir: str
    :return: the directories already done and the ones to do
    :rtype: Tuple[List[str], List[str]]
    """

    already_done = []
    to_do = []
    for dir_to_process in dirs_to_process:
        # existst and is not empty
        step_path = os.path.join(dir_to_process, step_dir)
        if os.path.isdir(step_path) and os.listdir(step_path):
            already_done.append(dir_to_process)
        else:
            to_do.append(dir_to_process)
    return already_done, to_do

def check_file(dirs_to_process: List[str], filepath: str) -> Tuple[List[str], List[str]]:
    """
    Check whether a file exists in a specific directory

    :param dirs_to_process: the base directories to look for the specific file
    :type dirs_to_process: List[str]
    :param filepath: the name of the specific file
    :type filepath: str
    :return: the directories already done and the ones to do
    :rtype: Tuple[List[str], List[str]]
    """

    already_done = []
    to_do = []
    for dir_to_process in dirs_to_process:
        # existst and is not empty
        step_path = os.path.join(dir_to_process, filepath)
        if os.path.isfile(step_path):
            already_done.append(dir_to_process)
        else:
            to_do.append(dir_to_process)
    return already_done, to_do

def get_dir_name(parameters: Dict[str, str], previous:bool = False) -> str:
    """
    Get a directory name from a list of parameters used to run the algorithm

    :param parameters: the dictionary where the information about the run is
    :param previous: True if you need to have the past directory information before
    applying NST or histogram matching
    :return: the name of the directory
    """
    dir_name = ""
    method = parameters["method"]
    replace = parameters["replace"]
    range_method = parameters["range_method"]
    special = parameters["special"]
    correct = parameters["correct"]

    if method:
        dir_name += "_" + method
    if replace:
        dir_name += "_" + replace
    if range_method:
        dir_name += "_" + range_method
    if not previous:
        if special:
            dir_name += "_" + special
    if correct:
        dir_name += "_" + correct

    if not dir_name:
        dir_name = FIRST_RUN_DIR_NAME
    else:
        dir_name = dir_name[1:]

    return dir_name

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
        except NotADirectoryError:
            # thumbs.db file
            pass


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

def check_subject_folder_structure(folder_path: str) -> dict:
    """
    Check if a folder either:
    1. Contains one or multiple folders beginning with Subject\d+
    2. Is itself a child/grandchild of a directory named Subject\d+
    
    :param folder_path: Path to the folder to check
    :type folder_path: str
    :return: Dictionary with structure information
    :rtype: dict
    """
    import os
    import re
    from pathlib import Path
    
    result = {
        "contains_subject_folders": False,
        "subject_folders": [],
        "is_within_subject_folder": False,
        "parent_subject_folder": None,
        "folder_structure": "unknown"
    }
    
    # Convert to Path object for easier manipulation
    folder_path = Path(folder_path)
    
    # Check if the folder contains Subject folders
    if folder_path.exists() and folder_path.is_dir():
        # Get all subdirectories
        subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
        
        # Check for Subject\d+ pattern
        subject_folders = [d for d in subdirs if re.match(r'Subject\d+', d.name, re.IGNORECASE)]
        
        if subject_folders:
            result["contains_subject_folders"] = True
            result["subject_folders"] = [str(folder) for folder in subject_folders]
            result["folder_structure"] = "parent"
    
    # Check if the folder is within a Subject folder
    parent_path = folder_path.parent
    grandparent_path = parent_path.parent if parent_path else None
    
    # Check if parent is a Subject folder
    if parent_path and re.match(r'Subject\d+', parent_path.name, re.IGNORECASE):
        result["is_within_subject_folder"] = True
        result["parent_subject_folder"] = str(parent_path)
        result["folder_structure"] = "child"
    
    # Check if grandparent is a Subject folder
    elif grandparent_path and re.match(r'Subject\d+', grandparent_path.name, re.IGNORECASE):
        result["is_within_subject_folder"] = True
        result["parent_subject_folder"] = str(grandparent_path)
        result["folder_structure"] = "grandchild"
    
    return result

