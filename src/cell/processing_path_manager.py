from dataclasses import dataclass
import os
from pathlib import Path
import re
import pickle

from src.configs.parser import Parser
from src.cell.cell_detection.mdrnn.mdrnn_constants import (ANALYSIS_DIR, FIRST_RUN_DIR_NAME,
                        MDRNN_CONE_DETECTOR_OUT_DIR,
                        MDRNN_CONE_DETECTOR_PREP_DIR, MDRNN_LUT_FILENAME,
                        OUTPUT_DIR, RAW_DATA_DIR_NAME, PLOT_DIR, RAW_RUN,
                        build_parameters_best_method,
                        build_parameters_best_preprocessing, VISUALIZE_PATCHES)
from src.shared.helpers.os_helpers import empty_folder

@dataclass
class ProcessingPathManager:
    """
    Manages the processing paths for different analysis types.

    :param path: The base path for processing.
    :type path: Path
    """

    def __init__(self, path: Path):
        """
        Initialize the ProcessingPathManager with the given path.

        :param path: The base path for processing.
        :type path: Path
        """
        Parser.initialize()
        self.path = path
        self.montage = self.Montage(self)
        self.mdrnn = self.MDRNN(self)
        self.density = self.Density(self)
        self.diabetic = self.Diabetic(self)
        self.register = self.Register(self)
        self.atms = self.ATMS(self)
        try:
            self.subject = re.search('Subject\d+', str(path)).group()
            self.subject_id = int(re.search('\d+', self.subject).group())
            self.session = re.search('Session\d+', str(path)).group()
            self.session_id = int(re.search('\d+', self.session).group())
        except AttributeError:
            self.subject = None
            self.subject_id = None
            self.session = None
            self.session_id = None
        try:
            self.right_eye = '_OD_' in next(self.path.glob(f'Subject{self.subject_id}_Session{self.session_id}*.tif')).name
        except StopIteration:
            self.right_eye = None

    def get_montage_mosaic(self):
        """
        Get the montage mosaic path.

        :return: The montage mosaic path.
        :rtype: Path
        """
        with open(str(self.montage.corrected_path / 'mosaic.pkl'), 'rb') as f:
            return pickle.load(f)

    class Montage:
        """
        Manages the montage processing paths.
        """

        def __init__(self, outer):
            """
            Initialize the Montage manager.

            :param outer: The outer ProcessingPathManager instance.
            :type outer: ProcessingPathManager
            """
            self.outer = outer
            self.initialize_montage()

        def initialize_montage(self):
            """
            Initialize the montage paths and settings.
            """
            self.path = self.outer.path / Parser.get_montaged_dir()
            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)
            self.gui_location_file = Parser.get_gui_done()
            self.corrected_path = self.outer.path / Parser.get_corrected_dir()
            self.parallel = Parser.run_ssim_parallel()
            self.n = Parser.get_ssim_n()
            self.ssim_path = self.outer.path / Parser.get_ssim_dir()
            self.shape_ratio = None

        def clean_montage(self):
            """
            Clean the montage directories by ensuring they exist.
            """
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
            if not os.path.isdir(self.corrected_path):
                os.makedirs(self.corrected_path)
            if not os.path.isdir(self.ssim_path):
                os.makedirs(self.ssim_path)

    class MDRNN:
        """
        Manages the MDRNN processing paths.
        """

        def __init__(self, outer):
            """
            Initialize the MDRNN manager.

            :param outer: The outer ProcessingPathManager instance.
            :type outer: ProcessingPathManager
            """
            self.outer = outer
            self.initialize_mdrnn()

        def initialize_mdrnn(self):
            """
            Initialize the MDRNN paths and settings.
            """
            self.analysis_dir = ANALYSIS_DIR
            self.lut_filename = MDRNN_LUT_FILENAME
            self.plot_dir = PLOT_DIR
            self.plot_path = self.outer.path / self.plot_dir
            self.output_dir = OUTPUT_DIR
            self.output_path = self.outer.path / self.output_dir
            os.makedirs(self.output_path, exist_ok=True)
            self.analysis_path = self.output_path / self.analysis_dir
            self.first_run_dir = FIRST_RUN_DIR_NAME
            self.first_run_path = self.outer.path / self.first_run_dir
            self.raw_run_dir = RAW_DATA_DIR_NAME
            self.raw_run_path = self.output_path / self.raw_run_dir
            self.run_nst = Parser.do_nst()

        def clean_mdrnn(self):
            """
            Clean the MDRNN directories by ensuring they exist or are emptied.
            """
            if os.path.isdir(self.output_path):
                empty_folder(self.output_path)
            else:
                os.makedirs(self.output_path)

    class ATMS:
        """
        Manages the ATMS processing paths.
        """
        def __init__(self, outer):
            """
            Initialize the MDRNN manager.

            :param outer: The outer ProcessingPathManager instance.
            :type outer: ProcessingPathManager
            """
            self.outer = outer
            self.initialize_atms()

        def initialize_atms(self):
            """
            Initialize the ATMS paths and settings.
            """
            self.path = self.outer.path / Parser.get_output_dir_atms()

    class Register:
        """
        Manages the registration processing paths.
        """

        def __init__(self, outer):
            """
            Initialize the Register manager.

            :param outer: The outer ProcessingPathManager instance.
            :type outer: ProcessingPathManager
            """
            self.outer = outer
            self.initialize_register()

        def initialize_register(self):
            """
            Initialize the registration paths.
            """
            self.path = self.outer.path / Parser.get_registring_dir()
            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)

    class Density:
        """
        Manages the density analysis processing paths.
        """

        def __init__(self, outer):
            """
            Initialize the Density manager.

            :param outer: The outer ProcessingPathManager instance.
            :type outer: ProcessingPathManager
            """
            self.outer = outer
            self.initialize_density()

        def initialize_density(self):
            """
            Initialize the density analysis paths.
            """
            self.path = self.outer.path / Parser.get_density_analysis_dir()
            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)
            self.layer_path = self.outer.path / Parser.get_layer_thickness_dir()

    class Diabetic:
        """
        Manages the diabetic analysis processing paths.
        """

        def __init__(self, outer):
            """
            Initialize the Diabetic manager.

            :param outer: The outer ProcessingPathManager instance.
            :type outer: ProcessingPathManager
            """
            self.outer = outer
            self.initialize_diabetic()

        def initialize_diabetic(self):
            """
            Initialize the diabetic analysis paths.
            """
            self.path = self.outer.path / Parser.get_diabetic_analysis_dir()
            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)
