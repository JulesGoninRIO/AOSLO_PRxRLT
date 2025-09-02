from typing import List, Dict
from pathlib import Path
import os
import csv
import numpy as np
import sys

from src.shared.datafile.datafile_constants import ImageModalities
from src.shared.datafile.image_file import ImageFile
from src.shared.datafile.datafile import DataFile
from src.configs.parser import Parser
from src.cell.affine_transform import AffineTransform
from src.cell.registration.registration_strategy import RegistrationStrategy, POCRegistrationStrategy, ECCRegistrationStrategy, MIRegistrationStrategy
from src.shared.computer_vision.image import get_blended_image
from src.shared.helpers.metrics import zero_normalized_cross_correlation, normalized_cross_correlation, nmi, normalized_mean_square_error, ssim, mutual_information
from src.shared.datafile.helpers import find_files, find_corresponding_files
from src.cell.processing_path_manager import ProcessingPathManager

class ImageRegistration:
    """
    A class to handle image registration processes.

    This class manages the registration strategies, modalities, and metrics for image registration.

    :param path_manager: The path manager for handling file paths.
    :type path_manager: ProcessingPathManager
    :param do_analysis: Flag to indicate whether to perform analysis.
    :type do_analysis: bool
    """
    def __init__(self, path_manager: ProcessingPathManager, do_analysis: bool = False):
        self.path_manager = path_manager
        # os.makedirs(self.register_path, exist_ok=True)
        self.__csv_name = Parser.get_registring_csv_name()
        self.__do_analysis = do_analysis  # Parser.do_registring_analysis()
        self.__set_strategies_and_modalities()
        self.__initialize_total_warp()
        self.metrics = self.__initialize_metrics() if self.__do_analysis else None

    def __set_strategies_and_modalities(self):
        """
        Set the registration strategies and modalities.

        This method sets the registration strategies and modalities based on whether analysis is to be performed.
        """
        if self.__do_analysis:
            self.__strategies = {
                "ECC": ECCRegistrationStrategy(),
                "MI": MIRegistrationStrategy(),
                "POC": POCRegistrationStrategy()
            }
            self.__modalities = [ImageModalities.CO, ImageModalities.CS, ImageModalities.DF]
        else:
            self.__strategies = {"POC": POCRegistrationStrategy()}
            self.__modalities = [ImageModalities.CO]

    def __initialize_total_warp(self):
        """
        Initialize the total warp dictionary.

        This method initializes the total warp dictionary for each strategy and modality.
        """
        self.total_warp = {strategy_key: {modality.value: [] for modality in self.__modalities} for strategy_key in self.__strategies}

    def __initialize_metrics(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Initialize the metrics dictionary.

        This method initializes the metrics dictionary for each strategy and modality.

        :return: The initialized metrics dictionary.
        :rtype: Dict[str, Dict[str, List[float]]]
        """
        metrics = {
            metric: {
                strategy_key: {modality.value: [] for modality in self.__modalities}
                for strategy_key in self.__strategies
            } for metric in ["ssims", "mis", "nmses", "nccs", "nmis", "znccs"]
        }
        return metrics

    def run_registration(self) -> Dict[str, Dict[str, List[List]]]:
        """
        Run the image registration process.

        This method retrieves the images to register, performs the registration, and returns the total warp results.

        :return: A dictionary containing the total warp results for each strategy and modality.
        :rtype: Dict[str, Dict[str, List[List]]]
        """
        images_to_register = self.__get_registration_images()
        self.__register_images(images_to_register)
        # df_results = self.registration_plots()
        # df_results.to_pickle(os.path.join(self.register_path, "df_results.pickle"))
        return self.total_warp

    def __get_registration_images(self) -> Dict[DataFile, List[DataFile]]:
        """
        Get the images to register.

        This method finds the optical absorption (OA) images and their corresponding modality images.

        :return: A dictionary mapping OA images to their corresponding modality images.
        :rtype: Dict[DataFile, List[DataFile]]
        """
        oa_images = find_files(self.path_manager.path, ImageFile, modalities=ImageModalities.OA)
        images_to_register = find_corresponding_files(self.path_manager.path, oa_images, self.__modalities)
        return images_to_register

    def __register_images(self, images_to_register: Dict[DataFile, List[DataFile]]) -> None:
        """
        Register the images.

        This method reads the data for each OA image and its corresponding modality images, and processes each pair.

        :param images_to_register: A dictionary mapping OA images to their corresponding modality images.
        :type images_to_register: Dict[DataFile, List[DataFile]]
        """
        for oa_image, modalities_present in images_to_register.items():
            oa_image.read_data(self.path_manager.path)
            for modality_image in modalities_present:
                modality_image.read_data(self.path_manager.path)
                self.__process_image(oa_image, modality_image)
        # self.__write_results()

    def __process_image(self, oa_image: np.ndarray, modality_image: np.ndarray):
        """
        Process a pair of images.

        This method registers the modality image to the OA image using each strategy and stores the results.

        :param oa_image: The OA image as a numpy array.
        :type oa_image: np.ndarray
        :param modality_image: The modality image as a numpy array.
        :type modality_image: np.ndarray
        """
        for strategy_key, strategy in self.__strategies.items():
            dx, dy, match_height = strategy.register(modality_image.data, oa_image.data)
            self.total_warp[strategy_key][modality_image.modality].append([modality_image, dx, dy, match_height])
            if self.__do_analysis:
                self.__perform_analysis(strategy_key, oa_image, modality_image, dx, dy)

    def __perform_analysis(self, strategy_key: str, oa_image: np.ndarray, modality_image: np.ndarray, dx: int, dy: int):
        """
        Perform analysis on the registered images.

        This method computes the affine transformation, blends the images, and computes the metrics.

        :param strategy_key: The key of the registration strategy used.
        :type strategy_key: str
        :param oa_image: The OA image as a numpy array.
        :type oa_image: np.ndarray
        :param modality_image: The modality image as a numpy array.
        :type modality_image: np.ndarray
        :param dx: The x-axis translation.
        :type dx: int
        :param dy: The y-axis translation.
        :type dy: int
        """
        H = AffineTransform.from_translation(dx, dy)
        blended, boundaries = get_blended_image(H, modality_image.data, oa_image.data)
        self.__compute_metrics(strategy_key, modality_image.modality, boundaries, oa_image.data, modality_image.data)

    def write_results(self):
        """
        Write the registration results to a CSV file.

        This method writes the total warp results for the POC strategy and Confocal modality to a CSV file.
        """
        with open(os.path.join(self.path_manager.register.path, self.__csv_name), "w", newline='') as f:
            writer = csv.writer(f)
            for row in self.total_warp['POC']["Confocal"]:
                writer.writerow(row)

    def __compute_metrics(self, strategy: str, modality: ImageModalities, boundaries: List[int],
                        oa_image: np.ndarray, modality_image: np.ndarray) -> None:
        """
        Compute the following metrics for outliers detection:
            Zero-Normalized Cross-Correlation (ZNCC)
            Normalized Cross-Correlation (NCC)
            Normalized Mutual Information (NMI)
            Normalized Mean Square Error (NMSE)
            Structural Similarity Index Measure (SSIM)
            Mutual Information (MI)

        :param strategy: the strategy used
        :type strategy: str
        :param modality: the modality we are dealing with (other than OA850nm)
        :type modality: ImageModalities
        :param boundaries: the overlapping region [top, bottom, left, right]
                        of the OA850nm image
        :type boundaries: List[int]
        :param oa_image: the OA850nm image
        :type oa_image: np.ndarray
        :param modality_image: the other modality image registered
        :type modality_image: np.ndarray
        """

        top, bottom, left, right = boundaries
        assert top < bottom, "The top pixel is bigger than the bottom pixel"
        assert left < right, "The left pixel is bigger than the right pixel"

        # Cut images to compare their score only on the overlapping region
        template_resized = oa_image[top:bottom+1, left:right+1]
        image_resized = modality_image[top:bottom+1, left:right+1]

        assert image_resized.shape == template_resized.shape, "The resulting images are not the same size, should debug"

        metrics = {
            'ZNCC': zero_normalized_cross_correlation(image_resized, template_resized),
            'NCC': normalized_cross_correlation(image_resized, template_resized),
            'NMI': nmi(image_resized, template_resized),
            'NMSE': normalized_mean_square_error(image_resized, template_resized),
            'SSIM': ssim(image_resized, template_resized),
            'MI': mutual_information(np.histogram2d(modality_image.ravel(), oa_image.ravel(), bins=20)[0])
        }

        for metric_name, score in metrics.items():
            getattr(self, metric_name.lower() + 's')[strategy][modality].append(score)

if __name__ == "__main__":
    Parser.initialize()
    dir_to_process = Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\run\Subject104\Session492')
    path_manager = ProcessingPathManager(dir_to_process)
    register = ImageRegistration(path_manager, do_analysis = False)
    warps = register.run_registration()
    register.write_results()