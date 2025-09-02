from typing import List, Tuple
import logging
import csv
import os
import numpy as np
import cv2
import pickle
from pathlib import Path

import sys
sys.path.append(r'C:\Users\Kattenburg\Documents\aoslo_pipeline')

from src.cell.processing_path_manager import ProcessingPathManager
from src.cell.montage.AOAutomontagingPython.AutoAOMontaging import AutoAOMontaging
from src.cell.montage.constants import PYTHON_38_PATH, LUT_FILENAME, FUNDUS_FOLDER
from src.cell.montage.montage_mosaic import MontageMosaic
from src.cell.montage.montage_mosaic_builder import CorrectedMontageMosaicBuilder, MatlabMontageMosaicBuilder
#TODO: move all from Montaging
# from src.cell.montage.run_auto_ao_montaging import run_auto_ao_montaging
from src.configs.parser import Parser
from src.shared.datafile.datafile_constants import ImageModalities
from src.shared.datafile.image_file import ImageFile

class MontagePipelineManager:
    """
    Manages the montage pipeline process.

    This class handles the initialization and execution of the montage pipeline,
    including optional steps for running Matlab scripts and saving the results.

    :param processsing_path_manager: The path manager for processing paths.
    :type processsing_path_manager: ProcessingPathManager
    """
    def __init__(
        self, processsing_path_manager: ProcessingPathManager,
        do_montaging: bool = False, run_matlab: bool = False):
        """
        Initialize the MontagePipelineManager.

        This method initializes the path manager and sets up the montage parameters.

        :param processsing_path_manager: The path manager for processing paths.
        :type processsing_path_manager: ProcessingPathManager
        :param do_montaging: Flag to indicate whether montaging should be performed.
        :type do_montaging: bool
        :param run_matlab: Flag to indicate whether Matlab scripts should be run.
        :type run_matlab: bool
        """
        self.path_manager = processsing_path_manager
        self.do_montaging = do_montaging
        self.run_matlab = run_matlab
        self.lut_filename = LUT_FILENAME

    def run(self) -> MontageMosaic:
        """
        Run the montage pipeline.

        This method runs the montage pipeline, optionally executing Matlab scripts,
        and saves the resulting mosaic images.

        :return: The resulting montage mosaic.
        :rtype: MontageMosaic
        """
        logging.info(f"Running Montage step for {self.path_manager.path}")
        if self.do_montaging:
            self.create_lut_file()
            # See if we run Matlab or our implementation
            if self.run_matlab:
                # run with python 3.8
                path = str(self.path_manager.path)
                lut_filename = self.lut_filename
                montaged_path = str(self.path_manager.montage.path)
                function_code = f"""
import sys
import os
sys.path.append(r'C:\\Users\\BardetJ\\Downloads\\james\\DonwloaDS\\aoslo_pipeline-master')
from src.cell.montage.AOAutomontaging_master.AutoAOMontaging.for_redistribution_files_only.run_auto_ao_montaging import run_auto_ao_montaging
run_auto_ao_montaging(r'{path}', r'{lut_filename}', r'{montaged_path}')
                """
                python38_path = PYTHON_38_PATH
                import subprocess
                try:
                    result = subprocess.run([python38_path, "-c", function_code], capture_output=True, text=True)
                except FileNotFoundError:
                    logging.error(f"Python 3.8 not found at {python38_path}")
                    raise FileNotFoundError
                print(result)
                # run_auto_ao_montaging(self.path_manager.path, self.lut_filename, self.path_manager.montaged_path)
            else:
                AutoAOMontaging(self.path_manager.path, self.lut_filename, self.path_manager.montage.path)

        else:
            builder = CorrectedMontageMosaicBuilder(self.path_manager)# CorrectedMontageMosaicBuilder(self.path_manager.montaged_path)
            mosaic = builder.build_mosaic()
            mosaic.save()
            subject_n = self.path_manager.subject_id
            fundus_path = Path(FUNDUS_FOLDER)
            for eye in ['OD', 'OS', 'OG']:
                if (fundus_path / f'HRA_{eye}_{subject_n}.jpg').exists():
                    fundus = cv2.imread(str(fundus_path / f'HRA_{eye}_{subject_n}.jpg'), cv2.IMREAD_GRAYSCALE)
                    mosaic.save_on_fundus(fundus)
                    break
            else:
                logging.warning(f'Fundus image not found for Subject {subject_n}')
            mosaic.save_images()
            builder = None # for garbage collection
            mosaic.binary_map()

        return mosaic

    def create_lut_file(self) -> None:
        """
        Creates a csv file with the image names and their relative location
        from the fixation target
        """
        # Let's create the resulting obejects from the input directory
        confocal_image_files: List[str] = []
        cs_image_files: List[str] = []
        dark_image_files: List[str] = []
        logging.info(f'Reading data for {str(self.path_manager.path)}')
        for image_file in self.process_images():
            if image_file.modality == ImageModalities.CO.value:
                confocal_image_files.append(image_file)
            elif image_file.modality == ImageModalities.CS.value:
                cs_image_files.append(image_file)
            elif image_file.modality == ImageModalities.DF.value:
                dark_image_files.append(image_file)

        # Verify that we have the same number of Confocal, CalculatedSplit and
        # Dark Field images to have a proper montaging (no missing files)
        all_images_files, modalities = self.__get_all_files(confocal_image_files,
                                                            cs_image_files,
                                                            dark_image_files)

        # Write the resulting images taken for Montaging in the CSV file
        lut_data = self.__get_lut_data(all_images_files, modalities)

        # Write down the output csv file
        out_name = self.path_manager.montage.path / LUT_FILENAME
        with open(out_name, 'w', newline="") as f:
            write = csv.writer(f, delimiter=';')
            write.writerows(lut_data)

    def process_images(self):
        """
        Reads the image and write it in the correct format
        """
        for filename in self.path_manager.path.iterdir():
            try:
                image_file = ImageFile(filename.name)
            except ValueError:
                # It is not an image file
                continue

            image_file.read_data(self.path_manager.path)
            cv2.imwrite(str(self.path_manager.path / filename),
                        image_file.data.astype(np.uint8))
            image_file.erase_data()
            yield image_file

    def __get_all_files(self,
                        confocal_image_files: List[ImageFile],
                        cs_image_files: List[ImageFile],
                        dark_image_files: List[ImageFile]) -> Tuple[List[ImageFile],
                                                                    List[ImageModalities]]:
        """
        Get all the files that will be used to Montage the image of the retina.
        If some images from a modality are missing, we won't use this modality
        for the montage

        :param confocal_image_files: the list of Confocal images
        :type confocal_image_files: List[ImageFile]
        :param cs_image_files: the list of CalculatedSplit images
        :type cs_image_files: List[ImageFile]
        :param dark_image_files: the list of DarkField images
        :type dark_image_files: List[ImageFile]
        :return: the list of all images that will be use for Montaging
        :rtype: List[ImageFile]
        """

        modalities = []

        if len(confocal_image_files) < len(cs_image_files) or \
                len(confocal_image_files) < len(dark_image_files):
            logging.warning("We are lacking some Confocal images in the current "
                            f"{str(self.path_manager.path)} folder. As it is the main "
                            "modality used for Montaging this might result in a "
                            "unprecise montage.")
            if len(cs_image_files) > len(dark_image_files):
                all_images_files: List[ImageFile] = cs_image_files.copy()
                modalities.append("CalculatedSplit")
            elif len(dark_image_files) > len(cs_image_files):
                all_images_files: List[ImageFile] = dark_image_files.copy()
                modalities.append("DarkField")
            else:
                all_images_files: List[ImageFile] = cs_image_files.copy()
                all_images_files.extend(dark_image_files)
                modalities.append("DarkField")
                modalities.append("CalculatedSplit")
        else:
            all_images_files: List[ImageFile] = confocal_image_files.copy()
            modalities.append("Confocal")
            if len(confocal_image_files) > len(cs_image_files):
                logging.warning("One or more CalculatedSplit images are missing "
                                "compared to the Confocal images, will not use "
                                "this modality for montaging.")
                cs_image_files = None
            else:
                all_images_files.extend(cs_image_files)
                modalities.append("CalculatedSplit")
            if len(confocal_image_files) > len(dark_image_files):
                logging.warning("One or more DarkField images are missing "
                                "compared to the Confocal images, will not use "
                                "this modality for montaging.")
                dark_image_files = None
            else:
                all_images_files.extend(dark_image_files)
                modalities.append("DarkField")

        return all_images_files, modalities

    def __get_lut_data(self,
                       all_images_files: List[ImageFile],
                       modalities: List[ImageModalities]) -> np.array:
        """
        Gather the datafile informations that will be written in the csv file to
        guide the montaging and move files in the temporary montaging folder

        :param all_image_files: the Confocal images to montage, the CalculatedSplit
                                images to montage and the DarkField images to montage
        :type all_image_files: List[ImageFile]
        :param modalities: the modalities that will be use for montaging
        :type modalities: List[ImageModalities]
        :return: the data to write in the csv file in the format:
                 image name, , position in x-axis, position in y-axis, size of
                 the image in degrees
        :rtype: np.array
        """

        # The result is store in a list that will be then converted to an object array
        lut_data = []
        number_image_per_modality = int(len(all_images_files)/len(modalities))
        for ind, image_file in enumerate(all_images_files):
            if ind >= number_image_per_modality:
                break

            if image_file.subject_id < 49:
                # x axis is inversed
                x_position = -image_file.x_position
            else:
                x_position = image_file.x_position
            # Add the confocal images to write in the CSV file
            lut_data.append([f'{int(image_file.image_id)}',
                            f'{x_position}',
                             f'{image_file.y_position}',
                             f'{image_file.x_size}', ''])

        return np.array(lut_data, dtype=object)

# # Pipeline Stage Interface
# class PipelineStage:
#     def process(self, data):
#         raise NotImplementedError

# # Concrete Stages
# class StageOne(PipelineStage):
#     def process(self, data):
#         # Process data
#         print("Stage One processing")
#         return data + " processed by Stage One"

# class StageTwo(PipelineStage):
#     def process(self, data):
#         # Process data
#         print("Stage Two processing")
#         return data + ", processed by Stage Two"

# # Pipeline Manager
# class PipelineManager:
#     def __init__(self):
#         self.stages = []

#     def add_stage(self, stage):
#         self.stages.append(stage)

#     def run(self, data):
#         for stage in self.stages:
#             data = stage.process(data)
#         return data

# # Usage
# pipeline = PipelineManager()
# pipeline.add_stage(StageOne())
# pipeline.add_stage(StageTwo())

# result = pipeline.run("Data")
# print(result)

if __name__ == "__main__":
    from pathlib import Path
    # path_manager = ProcessingPathManager(Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject110\Session530'))
    path_manager = ProcessingPathManager(Path(r'D:\aoslo\data\Subject125\Session571'))
    manager = MontagePipelineManager(path_manager, do_montaging=True, run_matlab=True)
    manager.run()
    print("oe")