from typing import List, Tuple, Dict
import os
import csv
import cv2
from pathlib import Path
import re
import numpy as np
import sys
from src.cell.registration.image_registration import ImageRegistration
from src.cell.processing_path_manager import ProcessingPathManager
from src.cell.montage.montage_mosaic import MontageMosaic
from src.shared.computer_vision.voronoi import VoronoiDiagram
from src.configs.parser import Parser

# def register_images():
#     #TODO: add register class call + building the warps list
#     # load the results of the Registring step
#     warp_file = os.path.join(self.register_path, self.__csv_name)
#     with open(warp_file, "r") as f:
#         reader = csv.reader(f)
#         warps = list(reader)
#     max_width = int(np.max(np.abs(np.array(warps)[:, 2].astype(float))))
#     max_height = int(
#             np.max(np.abs(np.array(warps)[:, 1].astype(float))))

class ConeDrawingContext:
    pass
    # def __init__(self, strategy: DrawConesStrategy):
    #     self._strategy = strategy

    # def execute_draw(self):
    #     self._strategy.draw()

def get_drawing_strategy(strategy_type):
    return
    # strategies = {
    #     "confocal": DrawConesOnConfocal(),
    #     "calculated_split": DrawConesCalculatedSplit(),
    #     "confocal_voronoi": DrawConesOnConfocalWithVoronoi(),
    #     "calculated_split_voronoi": DrawConesCalculatedSplitWithVoronoi(),
    # }
    # return strategies.get(strategy_type, DrawConesOnConfocal())

# # Example usage
# strategy = get_drawing_strategy("confocal_voronoi")
# context = ConeDrawingContext(strategy)
# context.execute_draw()

# name of the directory where are found the images with the cones drawn
# for the Diabetic Analysis
DRAW_DIR = "draw"

# name of the directory where the resulting CSV file will be found
OUT_PATH = "out"

# name of the file where we save reuslts from Biomarker cones radius
RADIUS_FILE = "radius.csv"

# the patients that are diabetic in the analysis
DIABETIC_PATIENTS = ["Session488"]

# the file with the results of the diabetic analysis
GLOBAL_DIABETIC_FILE = "intensities.csv"

""" DENSITY ANALYSIS CONSTANTS """

# name of the directory where the comparison with normals will be found
NORMAL_DIR = "compare_to_normal"

# number of pixels from which if there are less pixel in the area to compute we
# will skip this area (to avoid having densities going to infinity)
AREA_THRESHOLD = 4819

# name of the file where the results from density and layer thickness of a subject lies
RESULT_NAME = "results.csv"

MONTAGE_IMAGE = "Confocal.tif"

class BiomarkerAnalyser:
    """
    A class to analyze biomarkers in medical images.

    :param path_manager: Manages paths for processing.
    :type path_manager: ProcessingPathManager
    :param montage_mosaic: Handles mosaic operations.
    :type montage_mosaic: MontageMosaic
    :param cones: Tuple containing lists of cone coordinates.
    :type cones: Tuple[List[List[int]]]
    """
    def __init__(self, path_manager: ProcessingPathManager, montage_mosaic: MontageMosaic, cones: Tuple[List[List[int]]]):
        self.path_manager = path_manager
        self.montage_mosaic = montage_mosaic
        self.cones = cones
        self.csv_name = Parser.get_registring_csv_name()
        self.draw_path = self.path_manager.diabetic.path / DRAW_DIR
        self.out_path = self.path_manager.diabetic.path / OUT_PATH
        self.radius_file = RADIUS_FILE

    def run(self) -> None:
        """
        Execute the main analysis workflow.
        """
        self.warps = self.load_warps()
        self.get_max_shifts()
        self.shape = self.montage_mosaic.get_shape()
        self.binary_map = self.montage_mosaic.binary_map()

        resulting_image, resulting_cs_image = self.process_warps()

        self.save_images(resulting_image, resulting_cs_image)
        self.process_voronoi(resulting_image, resulting_cs_image)

    def process_warps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process image warps to generate blended images.

        :return: Tuple containing resulting images for different modalities.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        for i, image_warp in enumerate(self.warps):
            confocal_name = image_warp[0]
            try:
                transform = self.montage_mosaic.get_transform(confocal_name).matrix.copy()
            except ValueError:
                continue

            resulting_image = self.process_modality(i, image_warp, confocal_name, transform, "Confocal")
            resulting_cs_image = self.process_modality(i, image_warp, confocal_name, transform, "CalculatedSplit")

        return resulting_image, resulting_cs_image

    def process_modality(self, i: int, image_warp: List[int], confocal_name: str, transform: np.ndarray, modality: str) -> np.ndarray:
        """
        Process a specific modality to generate a blended image.

        :param i: Index of the current warp.
        :type i: int
        :param image_warp: List of warp parameters.
        :type image_warp: List[int]
        :param confocal_name: Name of the confocal image.
        :type confocal_name: str
        :param transform: Transformation matrix.
        :type transform: np.ndarray
        :param modality: Modality type (e.g., "Confocal", "CalculatedSplit").
        :type modality: str
        :return: Blended image for the specified modality.
        :rtype: np.ndarray
        """
        oa_name = re.sub(modality, "OA850nm", confocal_name)
        oa_path = self.path_manager.path
        blended_output_name = re.sub(modality, f"{modality}_OA", confocal_name)
        blended_image = self.add_oa_to_connected(self.path_manager.path, confocal_name, oa_name, oa_path, blended_output_name, image_warp, transform)

        if i == 0:
            resulting_image = np.zeros(blended_image.shape).astype(np.uint8)
        resulting_image += blended_image

        self.draw_cones_on_images(blended_image, confocal_name, voronoi=False)
        self.draw_cones_on_images(blended_image, confocal_name, voronoi=True)

        return resulting_image

    def save_images(self, resulting_image: np.ndarray, resulting_cs_image: np.ndarray) -> None:
        """
        Save the resulting images to disk.

        :param resulting_image: Resulting image for the Confocal modality.
        :type resulting_image: np.ndarray
        :param resulting_cs_image: Resulting image for the CalculatedSplit modality.
        :type resulting_cs_image: np.ndarray
        """
        cv2.imwrite(str(self.path_manager.diabetic.path / "Confocal_OA.tif"), resulting_image)
        cv2.imwrite(str(self.path_manager.diabetic.path / "CalculatedSplit_OA.tif"), resulting_cs_image)

    def process_voronoi(self, resulting_image: np.ndarray, resulting_cs_image: np.ndarray) -> None:
        """
        Process Voronoi diagrams for the resulting images.

        :param resulting_image: Resulting image for the Confocal modality.
        :type resulting_image: np.ndarray
        :param resulting_cs_image: Resulting image for the CalculatedSplit modality.
        :type resulting_cs_image: np.ndarray
        """
        binary_map = self.montage_mosaic.binary_map()
        binary_map = np.pad(binary_map, ((self.max_height, self.max_height), (self.max_width, self.max_width)), 'constant', constant_values=0)

        self.create_voronoi(resulting_image, "Confocal_OA_voronoi.tif")
        self.create_voronoi(resulting_cs_image, "CalculatedSplit_OA_voronoi.tif")

    def create_voronoi(self, image: np.ndarray, output_name: str) -> None:
        """
        Create and save a Voronoi diagram for the given image.

        :param image: Image to process.
        :type image: np.ndarray
        :param output_name: Name of the output file.
        :type output_name: str
        """
        cones_in_image = self.get_cones_on_image(image)
        voronoi = VoronoiDiagram(cones_in_image, image)
        voronoi.draw(output_name, self.path_manager.diabetic.path)
        area_to_analyze_image = voronoi.get_radius(self.binary_map)

        with open(os.path.join(self.path_manager.diabetic.path, self.radius_file), "w", newline='') as f:
            writer = csv.writer(f)
            for center, radius in area_to_analyze_image.items():
                row = [[center, radius]]
                writer.writerow(row)

    def add_oa_to_connected(
        self,
        image_path: Path,
        confocal_name: str,
        oa_name: str,
        oa_path: Path,
        blended_output_name: str,
        warp: np.ndarray,
        transform: np.ndarray) -> np.ndarray:
        """
        Add OA image to the confocal image and save the blended result.

        :param image_path: Path to the confocal image.
        :type image_path: Path
        :param confocal_name: Name of the confocal image file.
        :type confocal_name: str
        :param oa_name: Name of the OA image file.
        :type oa_name: str
        :param oa_path: Path to the OA image.
        :type oa_path: Path
        :param blended_output_name: Name of the output blended image file.
        :type blended_output_name: str
        :param warp: Warp parameters for blending.
        :type warp: np.ndarray
        :param transform: Transformation matrix.
        :type transform: np.ndarray
        :return: Blended image.
        :rtype: np.ndarray
        """
        confocal_image_padded = self.load_and_pad_image(image_path, confocal_name, transform)
        oa_image = cv2.imread(str(oa_path / oa_name), cv2.IMREAD_COLOR)
        blended = self.blend_images(confocal_image_padded, oa_image, warp, transform)
        cv2.imwrite(str(self.draw_path / blended_output_name), blended)
        return blended

    def get_image_path(self, draw: bool, first: bool) -> str:
        """
        Get the path for the image based on the flags.

        :param draw: Flag to indicate drawing path.
        :type draw: bool
        :param first: Flag to indicate first image.
        :type first: bool
        :return: Path to the image.
        :rtype: str
        """
        if not first:
            return self.diabetic_analysis_path if draw else self.corrected_path
        return self.corrected_path

    def load_warps(self) -> list:
        """
        Load warp parameters from a CSV file.

        :return: List of warp parameters.
        :rtype: list
        """
        warp_file = os.path.join(self.path_manager.register.path, self.csv_name)
        with open(warp_file, "r") as f:
            reader = csv.reader(f)
            return list(reader)

    def get_max_shifts(self) -> tuple:
        """
        Get the maximum shifts in width and height from the warp parameters.

        :return: Tuple containing maximum width and height shifts.
        :rtype: tuple
        """
        self.max_width = int(np.max(np.abs(np.array(self.warps)[:, 2].astype(float))))
        self.max_height = int(np.max(np.abs(np.array(self.warps)[:, 1].astype(float))))

    def get_image_names(self, image_warp: list, modality: str, first: bool, draw: bool) -> tuple:
        """
        Get the names of the confocal and OA images based on the modality and flags.

        :param image_warp: List of warp parameters.
        :type image_warp: list
        :param modality: Modality type (e.g., "Confocal", "CalculatedSplit").
        :type modality: str
        :param first: Flag to indicate if it is the first image.
        :type first: bool
        :param draw: Flag to indicate if drawing is enabled.
        :type draw: bool
        :return: Tuple containing the confocal and OA image names.
        :rtype: tuple
        """
        confocal_name = image_warp[0]
        oa_name = re.sub(modality, "OA850nm", confocal_name)
        if modality not in confocal_name:
            confocal_name = re.sub("Confocal", modality, confocal_name)
        if first:
            oa_name = confocal_name[:-4] + "_voronoi_cones.tif"
        elif draw:
            confocal_name = confocal_name[:-4] + "_voronoi_cones_padded.tif"
        return confocal_name, oa_name

    def load_and_pad_image(self, image_path: Path, image_name: str, transform: np.ndarray) -> np.ndarray:
        """
        Load and pad the image based on the transformation matrix.

        :param image_path: Path to the image.
        :type image_path: Path
        :param image_name: Name of the image file.
        :type image_name: str
        :param transform: Transformation matrix.
        :type transform: np.ndarray
        :return: Padded and transformed image.
        :rtype: np.ndarray
        """
        image = cv2.imread(str(image_path / image_name), cv2.IMREAD_COLOR)
        image_transformed = cv2.warpAffine(image, transform, self.shape)
        return np.pad(image_transformed, ((self.max_height, self.max_height), (self.max_width, self.max_width), (0, 0)), 'constant', constant_values=0)

    def blend_images(self, confocal_image_padded: np.ndarray, oa_image_padded: np.ndarray, image_warp: list, transform: np.ndarray) -> np.ndarray:
        """
        Blend the confocal and OA images based on the warp parameters and transformation matrix.

        :param confocal_image_padded: Padded confocal image.
        :type confocal_image_padded: np.ndarray
        :param oa_image_padded: Padded OA image.
        :type oa_image_padded: np.ndarray
        :param image_warp: List of warp parameters.
        :type image_warp: list
        :param transform: Transformation matrix.
        :type transform: np.ndarray
        :return: Blended image.
        :rtype: np.ndarray
        """
        M = transform.copy()
        M[0, 2] += eval(image_warp[2])
        M[1, 2] += eval(image_warp[1])
        dst = cv2.warpAffine(oa_image_padded, M, self.shape)
        dst = np.pad(dst, ((self.max_height, self.max_height), (self.max_width, self.max_width), (0, 0)), 'constant', constant_values=0)
        return cv2.addWeighted(confocal_image_padded, 0.4, dst, 0.4, 0)

    def get_cones_on_image(self, blended_image: np.ndarray) -> List[List[int]]:
        """
        Get the coordinates of cones present in the blended image.

        :param blended_image: Blended image.
        :type blended_image: np.ndarray
        :return: List of coordinates of cones in the image.
        :rtype: List[List[int]]
        """
        cones_mdrnn, cones_atms = self.cones
        cones_in_image = []
        for cone_mdrnn in cones_mdrnn:
            if np.any(blended_image[cone_mdrnn[1], cone_mdrnn[0], :]):
                blended_image = cv2.circle(blended_image, (cone_mdrnn[0], cone_mdrnn[1]), radius=0, color=(255, 0, 0), thickness=3)
                cones_in_image.append([cone_mdrnn[1], cone_mdrnn[0]])
        for cone_atms in cones_atms:
            if np.any(blended_image[cone_atms[1], cone_atms[0], :]):
                blended_image = cv2.circle(blended_image, (cone_atms[0], cone_atms[1]), radius=0, color=(255, 0, 0), thickness=3)
                cones_in_image.append([cone_atms[1], cone_atms[0]])
        return cones_in_image

    def draw_cones_on_images(self, blended_image, name, voronoi: bool = False) -> None:
        """
        Draw cones on image and returns the Voronoi areas

        :param modality: _description_, defaults to "Confocal"
        :type modality: str, optional
        :param voronoi: _description_, defaults to False
        :type voronoi: bool, optional
        """
        print("oe")
        cones_in_image = self.get_cones_on_image(blended_image)
        out_name = os.path.splitext(name)[0] + "_cones.tif"
        if voronoi:
            try:
                VoronoiDiagram(cones_in_image, blended_image).draw(out_name, self.draw_path)
            except RuntimeError:
                pass
        else:
            cv2.imwrite(str(self.draw_path / out_name), blended_image)

if __name__ == "__main__":
    path_manager = ProcessingPathManager(Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\run\Subject104\Session492'))
    from src.cell.montage.montage_mosaic_builder import CorrectedMontageMosaicBuilder
    from src.cell.cell_detection.cone_gatherer import ConeGatherer
    # builder = CorrectedMontageMosaicBuilder(path_manager)
    # mosaic = builder.build_mosaic()
    import pickle
    with open(r'C:\Users\BardetJ\Downloads\mosaic.pickle', 'rb') as file:
        mosaic = pickle.load(file)
    # mosaic.save()
    # subject_n = 104
    # fundus = cv2.imread(os.path.join(r'P:\AOSLO\AST GUI\FundusWithCenter', f'HRA_OD_{subject_n}.jpg'), cv2.IMREAD_GRAYSCALE)
    # mosaic.save_on_fundus(fundus)
    # mosaic.save_images()
    # cones = ConeGatherer(path_manager, mosaic).get_cones_from_sources()
    with open(r'C:\Users\BardetJ\Downloads\cones.pickle', 'rb') as file:
        cones = pickle.load(file)
    BiomarkerAnalyser(path_manager, mosaic, cones).run()
