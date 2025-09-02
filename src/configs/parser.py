import configparser
import os

from src.shared.helpers.exceptions import (
    NoOptionError, NoSectionError, NotAbsolutePathError, PathDoesNotExistError
)
from .read_config import (add_config_option, read_config_line, read_list)
from src.shared.helpers.global_constants import (
    ATMS_OVERLAP_DISTANCE, MDRNN_OVERLAP_DISTANCE,
    MDRNN_OVERLAP_DISTANCE_SAME_IMAGE)
from src.shared.helpers.validators import (
    _ParameterType, _SupportedTypes, _validate_path, _validate_value
)

# I'm gonna punch whoever wrote this 
# -Gian

# All relevant parameters are in src/config/config.txt, change that to change what is used in the rest of the pipeline

class Parser():
    """
    Gets all the informations from config file with optional and needed options
    """

    # The configs file lies in the same folder as this script
    __configs_file = os.path.abspath(os.path.join(os.path.abspath(__file__), \
        '..', 'config.txt'))

    @staticmethod
    def initialize():
        """
        Static method to initialize the different config parameters
        """
        Parser.configs = configparser.RawConfigParser()
        Parser.configs.read(Parser.__configs_file)

        # If the configparser is empty it means the Config file has not been found
        if len(Parser.configs._sections) == 0:
            raise FileNotFoundError(Parser.__configs_file)
        # add path of the config file for error handling
        Parser.__configs = add_config_option(Parser.configs, Parser.__configs_file)

        # Get the mandatory options from the config file
        Parser.__base_dir = read_config_line(Parser.configs, "os", "__base_dir")
        _validate_value(_SupportedTypes.STRING, Parser.__base_dir, \
            "__base_dir", Parser.__configs_file)

        Parser.__powershell = read_config_line(Parser.configs, "os", "__powershell")
        _validate_value(_SupportedTypes.STRING, Parser.__powershell, \
            "__powershell", Parser.__configs_file)

        # Montaging
        try:
            Parser.__do_montaging = eval(read_config_line(Parser.configs, "Montaging", \
                "__do_montaging"))
        except NoSectionError:
            # if the section does not exist then we do not do montaging
            Parser.__do_montaging = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_montaging, "__do_montaging", \
            Parser.__configs_file)

        Parser.__montaging_GUI_done = eval(read_config_line(\
            Parser.configs, "Montaging", "__GUI_done"))
        _validate_value(_SupportedTypes.BOOL, Parser.__montaging_GUI_done, \
            "__GUI_done", Parser.__configs_file)
        
        Parser.__montaging_GUI_location_file = read_config_line(\
            Parser.configs, "Montaging", "__GUI_location_file")
        _validate_value(_SupportedTypes.STRING, Parser.__montaging_GUI_location_file, \
            "__GUI_location_file", Parser.__configs_file)

        Parser.__corrected_montage_filename = read_config_line(\
            Parser.configs, "Montaging", "__corrected_montage_filename")
        _validate_value(_SupportedTypes.STRING, Parser.__corrected_montage_filename, \
            "__corrected_montage_filename", Parser.__configs_file)
        Parser.__montaging_do_SSIM = eval(read_config_line(\
            Parser.configs, "Montaging", "__do_SSIM"))
        _validate_value(_SupportedTypes.BOOL, Parser.__montaging_do_SSIM, \
            "__do_SSIM", Parser.__configs_file)
        try:
            Parser.__SSIM_n = eval(read_config_line(\
                Parser.configs, "Montaging", "__SSIM_n"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.INT, Parser.__SSIM_n, \
            "__SSIM_n", Parser.__configs_file)
        try:
            Parser.__ssim_parallel = eval(read_config_line(\
                Parser.configs, "Montaging", "__parallel"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.BOOL, Parser.__ssim_parallel, \
            "__parallel", Parser.__configs_file)
        try:
            Parser.__montaging_montaged_dir = read_config_line(\
                Parser.configs, "Montaging", "__montaged_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__montaging_montaged_dir, \
            "__montaged_dir", Parser.__configs_file)
        try:
            Parser.__montaging_corrected_dir = read_config_line(\
                Parser.configs, "Montaging", "__corrected_dir")
        except NoOptionError:
            # should load from constants -> add _corrected_dir = montaged_corrected
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__montaging_corrected_dir, \
            "__corrected_dir", Parser.__configs_file)
        try:
            Parser.__montaging_ssim_dir = read_config_line(\
                Parser.configs, "Montaging", "__ssim_dir")
        except NoOptionError:
            # should load from constants -> add _corrected_dir = montaged_corrected
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__montaging_ssim_dir, \
            "__ssim_dir", Parser.__configs_file)

        # Registring
        try:
            Parser.__do_registring = eval(read_config_line(Parser.configs, "Registring", \
                "__do_registring"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            Parser.__do_registring = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_registring, "__do_registring", \
            Parser.__configs_file)

        Parser.__registring_do_analysis = eval(read_config_line(\
            Parser.configs, "Registring", "__do_analysis"))
        _validate_value(_SupportedTypes.BOOL, Parser.__registring_do_analysis, \
            "__do_analysis", Parser.__configs_file)
        Parser.__registring_csv_name = read_config_line(\
            Parser.configs, "Registring", "__csv_name")
        _validate_value(_SupportedTypes.STRING, Parser.__registring_csv_name, \
            "__csv_name", Parser.__configs_file)
        try:
            Parser.__register_dir = read_config_line(\
                Parser.configs, "Registring", "__register_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__register_dir, \
            "__register_dir", Parser.__configs_file)

        # CellDetection
        # MDRNN
        try:
            Parser.__do_mdrnn = eval(read_config_line(Parser.configs, "CellDetection", \
                "__do_mdrnn"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            Parser.__do_mdrnn = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_mdrnn, "__do_mdrnn", \
            Parser.__configs_file)
        
        try:
            Parser.__do_nst = eval(read_config_line(Parser.configs, "CellDetection", \
                "__do_nst"))
        except NoOptionError:
            # if the option does not exist then we do not do NST
            Parser.__do_nst = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_nst, "__do_nst", \
            Parser.__configs_file)

        Parser.__mdrnn_do_analysis = eval(read_config_line(\
            Parser.configs, "CellDetection", "__do_analysis"))
        _validate_value(_SupportedTypes.BOOL, Parser.__mdrnn_do_analysis, \
            "__do_analysis", Parser.__configs_file)
        
        # try:
        #     Parser.__output_dir_mdrnn = read_config_line(\
        #         Parser.configs, "CellDetection", "__output_dir_mdrnn")
        # except NoOptionError:
        #     # should load from constants -> add __montaged_dir = montaged
        #     pass
        # _validate_value(_SupportedTypes.STRING, Parser.__output_dir_mdrnn, \
        #     "__output_dir_mdrnn", Parser.__configs_file)

        Parser.__mdrnn_do_analysis = eval(read_config_line(\
            Parser.configs, "CellDetection", "__do_analysis"))
        _validate_value(_SupportedTypes.BOOL, Parser.__mdrnn_do_analysis, \
            "__do_analysis", Parser.__configs_file)
        Parser.__methods = read_list(Parser.configs, "CellDetection", "__methods")
        Parser.__replaces = read_list(Parser.configs, "CellDetection", "__replaces")
        Parser.__range_methods = read_list(Parser.configs, "CellDetection", "__range_methods")
        Parser.__specials = read_list(Parser.configs, "CellDetection", "__specials")
        Parser.__corrects = read_list(Parser.configs, "CellDetection", "__corrects")
        try:
            Parser.__treats = read_config_line(\
                Parser.configs, "CellDetection", "__treats")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__treats, \
            "__treats", Parser.__configs_file)
        try:
            Parser.__visualize_patches = eval(read_config_line(\
                Parser.configs, "CellDetection", "__visualize_patches"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.BOOL, Parser.__visualize_patches, \
            "__visualize_patches", Parser.__configs_file)
        try:
            Parser.__mdrnn_overlap_distance_same_image = eval(read_config_line(\
                Parser.configs, "CellDetection", "__mdrnn_overlap_distance_same_image"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            Parser.__mdrnn_overlap_distance_same_image = MDRNN_OVERLAP_DISTANCE_SAME_IMAGE
            pass
        _validate_value(_SupportedTypes.INT, Parser.__mdrnn_overlap_distance_same_image, \
            "__mdrnn_overlap_distance_same_image", Parser.__configs_file)
        try:
            Parser.__mdrnn_overlap_distance = eval(read_config_line(\
                Parser.configs, "CellDetection", "__mdrnn_overlap_distance"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            Parser.__mdrnn_overlap_distance = MDRNN_OVERLAP_DISTANCE
            pass
        _validate_value(_SupportedTypes.INT, Parser.__mdrnn_overlap_distance, \
            "__mdrnn_overlap_distance", Parser.__configs_file)

        #ATMS
        try:
            Parser.__do_atms = eval(read_config_line(Parser.configs, "CellDetection", \
                "__do_atms"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            Parser.__do_atms = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_atms, "__do_atms", \
            Parser.__configs_file)
        try:
            Parser.__atms_overlap_distance = eval(read_config_line(\
                Parser.configs, "CellDetection", "__atms_overlap_distance"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            Parser.__atms_overlap_distance = ATMS_OVERLAP_DISTANCE
            pass
        _validate_value(_SupportedTypes.INT, Parser.__atms_overlap_distance, \
            "__atms_overlap_distance", Parser.__configs_file)
        try:
            Parser.__output_dir_atms = read_config_line(\
                Parser.configs, "CellDetection", "__output_dir_atms")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__output_dir_atms, \
            "__output_dir_atms", Parser.__configs_file)

        try:
            Parser.__axial_length_file = read_config_line(\
                Parser.configs, "CellDetection", "__axial_length_file")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__axial_length_file, \
            "__axial_length_file", Parser.__configs_file)

        #  Analysis
        try:
            Parser.__do_diabetic_analysis = eval(read_config_line(Parser.configs, \
                "Diabete", "__do_diabetic_analysis"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            Parser.__do_diabetic_analysis = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_diabetic_analysis, \
            "__do_diabetic_analysis", Parser.__configs_file)
        try:
            Parser.__diabetic_analysis_dir = read_config_line(\
                Parser.configs, "Diabete", "__diabetic_analysis_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__diabetic_analysis_dir, \
            "__diabetic_analysis_dir", Parser.__configs_file)

        try:
            Parser.__do_density_analysis = eval(read_config_line(Parser.configs, \
                "Density", "__do_density_analysis"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            Parser.__do_density_analysis = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_density_analysis, \
            "__do_density_analysis", Parser.__configs_file)
        try:
            Parser.__density_analysis_dir = read_config_line(\
                Parser.configs, "Density", "__density_analysis_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__density_analysis_dir, \
            "__density_analysis_dir", Parser.__configs_file)
        Parser.__density_growing_factor = read_list(Parser.configs, "Density", "__growing_factor")
        try:
            Parser.__density_center_limit = eval(read_config_line(\
                Parser.configs, "Density", "__center_limit"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.FLOAT, Parser.__density_center_limit, \
            "__center_limit", Parser.__configs_file)
        try:
            Parser.__density_curve_fit_limit = eval(read_config_line(\
                Parser.configs, "Density", "__curve_fit_limit"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.INT, Parser.__density_curve_fit_limit, \
            "__curve_fit_limit", Parser.__configs_file)
        try:
            Parser.__density_atms_limit = eval(read_config_line(\
                Parser.configs, "Density", "__atms_limit"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.INT, Parser.__density_atms_limit, \
            "__atms_limit", Parser.__configs_file)
        try:
            Parser.__density_mdrnn_limit = eval(read_config_line(\
                Parser.configs, "Density", "__mdrnn_limit"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.INT, Parser.__density_mdrnn_limit, \
            "__mdrnn_limit", Parser.__configs_file)
        try:
            Parser.__do_compare_to_layer_thickness = eval(read_config_line(\
                Parser.configs, "LayerThickness", "__do_compare_to_layer_thickness"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.BOOL, Parser.__do_compare_to_layer_thickness, \
            "__do_compare_to_layer_thickness", Parser.__configs_file)
        try:
            Parser.__layer_thickness_dir = read_config_line(\
                Parser.configs, "LayerThickness", "__layer_thickness_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__layer_thickness_dir, \
            "__layer_thickness_dir", Parser.__configs_file)

        try:
            Parser.__do_global_density = eval(read_config_line(\
                Parser.configs, "Density", "__do_global_analysis"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.BOOL, Parser.__do_global_density, \
            "__do_global_analysis", Parser.__configs_file)

        try:
            Parser.__global_density_dir = read_config_line(\
                Parser.configs, "Density", "__global_density_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__global_density_dir, \
            "__global_density_dir", Parser.__configs_file)
        try:
            Parser.__global_diabetic_dir = read_config_line(\
                Parser.configs, "Diabete", "__global_diabetic_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__global_diabetic_dir, \
            "__global_diabetic_dir", Parser.__configs_file)

        try:
            Parser.__do_global_diabetic = eval(read_config_line(\
                Parser.configs, "Diabete", "__do_global_analysis"))
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.BOOL, Parser.__do_global_diabetic, \
            "__do_global_analysis", Parser.__configs_file)

        # Blood Flow
        try:
            Parser.__do_blood_flow = eval(read_config_line(Parser.configs, \
                "BloodFlow", "__do_blood_flow"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            Parser.__do_blood_flow = False
        _validate_value(_SupportedTypes.BOOL, Parser.__do_blood_flow, \
            "__do_blood_flow", Parser.__configs_file)
        try:
            Parser.__blood_speed_estimation = eval(read_config_line(Parser.configs, \
                "BloodFlow", "__blood_speed_estimation"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            # Parser.__blood_vessel_tool = False
            pass
        _validate_value(_SupportedTypes.BOOL, Parser.__blood_speed_estimation, \
            "__blood_vessel_tool", Parser.__configs_file)
        try:
            Parser.__blood_model = eval(read_config_line(Parser.configs, \
                "BloodFlow", "__blood_model"))
        except NoSectionError:
            # if the section does not exist then we do not do registring
            # Parser.__blood_vessel_tool = False
            pass
        _validate_value(_SupportedTypes.BOOL, Parser.__blood_model, \
            "__blood_model", Parser.__configs_file)
        Parser.__vessel_size = read_list(Parser.configs, "BloodFlow", "__vessel_size")
        try:
            Parser.__speed_output_dir = read_config_line(\
                Parser.configs, "BloodFlow", "__speed_output_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__speed_output_dir, \
            "__speed_output_dir", Parser.__configs_file)
        try:
            Parser.__model_output_dir = read_config_line(\
                Parser.configs, "BloodFlow", "__model_output_dir")
        except NoOptionError:
            # should load from constants -> add __montaged_dir = montaged
            pass
        _validate_value(_SupportedTypes.STRING, Parser.__model_output_dir, \
            "__model_output_dir", Parser.__configs_file)

    @staticmethod
    def get_base_dir():
        """
        Static method to get the base directory of the PostProcessing Pipeline
        """
        return Parser.__base_dir

    # @staticmethod
    # def get_other_file_dir():
    #     """
    #     Static method to get the directory where to put the non-used modality files
    #     """
    #     return Parser.__other_file_dir

    # @staticmethod
    # def get_modalities():
    #     """
    #     Static method to get the modalities we will use in the pipeline
    #     """
    #     return Parser.__modalities

    @staticmethod
    def get_powershell():
        """
        Static method to get whether we are doing the montaging step or not
        """
        return Parser.__powershell

    @staticmethod
    def do_montaging():
        """
        Static method to get whether we are doing the montaging step or not
        """
        return Parser.__do_montaging

    @staticmethod
    def get_corrected_montage_filename():
        """
        Static method to get whether we are doing the montaging analysis
        """
        return Parser.__corrected_montage_filename

    @staticmethod
    def get_montaged_dir():
        """
        Static method to get the montage directory
        """
        return Parser.__montaging_montaged_dir

    @staticmethod
    def get_gui_done():
        """
        Static method to get if the GUI correction has been made
        """
        return Parser.__montaging_GUI_done

    @staticmethod
    def get_gui_location_file():
        """
        Static method to get the location of the GUI file
        """
        return Parser.__montaging_GUI_location_file

    @staticmethod
    def get_corrected_dir():
        """
        Static method to get the montage corrected directory
        """
        return Parser.__montaging_corrected_dir

    @staticmethod
    def do_ssim():
        """
        Static method to get whether to do ssim correction
        """
        return Parser.__montaging_do_SSIM

    @staticmethod
    def get_ssim_n():
        """
        Static method to get the number of pixels to look around with SSIM
        """
        return Parser.__SSIM_n

    @staticmethod
    def run_ssim_parallel():
        """
        Static method to get whether we will run SSIM in parallel
        """
        return Parser.__ssim_parallel

    @staticmethod
    def get_ssim_dir():
        """
        Static method to get the montage ssim directory
        """
        return Parser.__montaging_ssim_dir

    @staticmethod
    def do_registring():
        """
        Static method to get whether we are doing the registring or not
        """
        return Parser.__do_registring

    @staticmethod
    def get_registring_csv_name():
        """
        Static method to get whether we are doing the registring analysis or not
        """
        return Parser.__registring_csv_name

    @staticmethod
    def do_registring_analysis():
        """
        Static method to get whether we are doing the registring analysis or not
        """
        return Parser.__registring_do_analysis

    @staticmethod
    def get_registring_dir():
        """
        Static method to get the registring directory
        """
        return Parser.__register_dir

    @staticmethod
    def do_mdrnn_cell_detection():
        """
        Static method to get whether or not we are running MDRNN cell detection
        """
        return Parser.__do_mdrnn

    @staticmethod
    def do_nst():
        """
        Static method to get whether or not we are running NST
        """
        return Parser.__do_nst

    @staticmethod
    def do_mdrnn_analysis():
        """
        Static method to get whether or not we are running MDRNN cell detection
        """
        return Parser.__mdrnn_do_analysis

    @staticmethod
    def do_atms_cell_detection():
        """
        Static method to get whether or not we are running ATMS cell detection
        """
        return Parser.__do_atms

    @staticmethod
    def get_methods():
        """
        Static method to get the methods to run MDRNN cell detection
        """
        return Parser.__methods

    @staticmethod
    def get_replaces():
        """
        Static method to get the replaces to run MDRNN cell detection
        """
        return Parser.__replaces

    @staticmethod
    def get_range_methods():
        """
        Static method to get the range_methods to run MDRNN cell detection
        """
        return Parser.__range_methods

    @staticmethod
    def get_specials():
        """
        Static method to get the specials to run MDRNN cell detection
        """
        return Parser.__specials

    @staticmethod
    def get_corrects():
        """
        Static method to get the corrects to run MDRNN cell detection
        """
        return Parser.__corrects

    @staticmethod
    def get_treats():
        """
        Static method to get the treats to run MDRNN cell detection
        """
        return Parser.__treats

    @staticmethod
    def get_visualize_patches():
        """
        Static method to get the visualize_patches to run MDRNN cell detection
        """
        return Parser.__visualize_patches

    @staticmethod
    def get_mdrnn_overlap_distance_same_image():
        """
        Static method to get the MDRNN cell detection overlap distance on same image
        """
        return Parser.__mdrnn_overlap_distance_same_image

    @staticmethod
    def get_mdrnn_overlap_distance():
        """
        Static method to get the MDRNN cell detection overlap distance
        """
        return Parser.__mdrnn_overlap_distance

    @staticmethod
    def get_atms_overlap_distance():
        """
        Static method to get the ATMS cell detection overlap distance
        """
        return Parser.__atms_overlap_distance

    @staticmethod
    def get_output_dir_atms():
        """
        Static method to get the name of the directory for ATMS cell detection
        """
        return Parser.__output_dir_atms

    # @staticmethod
    # def get_output_dir_mdrnn():
    #     """
    #     Static method to get the name of the directory for MDRNN cell detection
    #     """
    #     return Parser.__output_dir_mdrnn

    @staticmethod
    def get_axial_length_file():
        """
        Static method to get the name of the axial length file
        """
        return Parser.__axial_length_file

    @staticmethod
    def do_diabetic_analysis():
        """
        Static method to get whether we are doing the diabetic analysis or not
        """
        return Parser.__do_diabetic_analysis

    @staticmethod
    def get_diabetic_analysis_dir():
        """
        Static method to get the directory of the diabetic analyis
        """
        return Parser.__diabetic_analysis_dir

    @staticmethod
    def do_density_analysis():
        """
        Static method to get whether we are doing the density analysis or not
        """
        return Parser.__do_density_analysis

    @staticmethod
    def get_density_growing_factor():
        """
        Static method to get the density growing factor
        """
        return Parser.__density_growing_factor

    @staticmethod
    def get_density_center_limit():
        """
        Static method to get the density center limit
        """
        return Parser.__density_center_limit

    @staticmethod
    def get_density_curve_fit_limit():
        """
        Static method to get the density curve fit limit
        """
        return Parser.__density_curve_fit_limit

    @staticmethod
    def get_density_curve_atms_limit():
        """
        Static method to get the density ATMS limit
        """
        return Parser.__density_atms_limit

    @staticmethod
    def get_density_curve_mdrnn_limit():
        """
        Static method to get the density MDRNN limit
        """
        return Parser.__density_mdrnn_limit

    @staticmethod
    def do_compare_to_layer_thickness():
        """
        Static method to know whether to do or not the comparison with layer thickness
        """
        return Parser.__do_compare_to_layer_thickness

    @staticmethod
    def get_density_analysis_dir():
        """
        Static method to get the directory of the density analysis
        """
        return Parser.__density_analysis_dir

    @staticmethod
    def get_layer_thickness_dir():
        """
        Static method to get the directory of the layer thickness analysis
        """
        return Parser.__layer_thickness_dir

    def do_global_density():
        """
        Static method to get the directory of the density analysis
        """
        return Parser.__do_global_density

    def do_global_diabetic():
        """
        Static method to get the directory of the density analysis
        """
        return Parser.__do_global_diabetic

    def get_global_diabetic_dir():
        """
        Static method to get the directory of the diabetic analysis
        """
        return Parser.__global_diabetic_dir

    def get_global_density_dir():
        """
        Static method to get the directory of the density analysis
        """
        return Parser.__global_density_dir

    @staticmethod
    def do_blood_flow():
        """
        Static method to get whether we are doing the Blood Flow analysis or not
        """
        return Parser.__do_blood_flow

    @staticmethod
    def do_blood_speed_estimation():
        """
        Static method to get whether we are doing the Blood Flow analysis or not
        """
        return Parser.__blood_speed_estimation

    @staticmethod
    def do_blood_model():
        """
        Static method to get whether we are doing the Blood Flow analysis or not
        """
        return Parser.__blood_model

    @staticmethod
    def get_speed_output_dir():
        """
        Static method to get the output directory for Blood Flow analysis
        """
        return Parser.__speed_output_dir

    @staticmethod
    def get_vessel_size():
        """
        Static method to get the output directory for Blood Flow analysis
        """
        return Parser.__vessel_size

    @staticmethod
    def get_model_output_dir():
        """
        Static method to get the output directory for Blood Flow analysis
        """
        return Parser.__model_output_dir

    @staticmethod
    def cell_detection_analysis():
        """
        Static method to initialize the analysis
        """
        # Parser.configs = configparser.RawConfigParser()
        # Parser.configs.read(Parser.__configs_file)

        methods = Parser.get_methods()
        replaces = Parser.get_replaces()
        range_methods = Parser.get_range_methods()
        specials = Parser.get_specials()
        corrects = Parser.get_corrects()
        treats = Parser.get_treats()

        return methods, replaces, range_methods, specials, corrects, treats

    # @staticmethod
    # def set_base_dir(value: str) -> None:
    #     """
    #     Static method to set the base directory

    #     :param value: the string with the path of the folder where the data is
    #     """
    #     Parser.base_dir = value
    #     # Parser.configs.set('CellDetection', '__base_dir', value)

    # # @staticmethod
    # # def get_common_input_dir() -> str:
    # #     """
    # #     Static method to get the  the base directory

    # #     :param value: the string with the path of the folder where the data is
    # #     """
    # #     return Parser.configs.get('CellDetection', '__common_input_dir')

    # @staticmethod
    # def get_output_dir_mdrnn() -> str:
    #     """
    #     Static method to get I/O paths to be subfolders of base_dir

    #     :return: the string Path of the output directory
    #     """
    #     return Parser.configs.get('CellDetection', '__output_dir_mdrnn')

    # @staticmethod
    # def get_output_dir_atms() -> str:
    #     """
    #     Static method to get I/O paths to be subfolders of base_dir

    #     :return: the string Path of the output directory
    #     """
    #     return Parser.configs.get('CellDetection', '__output_dir_atms')

    # @staticmethod
    # def do_montaging() -> bool:
    #     # whether we have to do the montaging step
    #     return eval(Parser.configs.get('Montaging', '__do_montaging'))

    # @staticmethod
    # def do_registring() -> bool:
    #     # whether we have to do the registring step
    #     return eval(Parser.configs.get('Registring', '__do_registring'))

    # @staticmethod
    # def do_diabetic_analysis() -> bool:
    #     # whether we have to do the diabetic analysis step
    #     return eval(Parser.configs.get('Analysis', '__do_diabetic_analysis'))

    # @staticmethod
    # def do_blood_flow() -> bool:
    #     # whether we have to do the blood flow analyis
    #     return eval(Parser.configs.get('BloodFlow', '__do_blood_flow'))

    # @staticmethod
    # def do_atms_cell_detection() -> bool:
    #     # whether we have to do the ATMS cell detection step
    #     return eval(Parser.configs.get('CellDetection', '__do_atms'))

    # @staticmethod
    # def do_mdrnn_cell_detection() -> bool:
    #     # whether we have to do the MDRNN cell detection step
    #     return eval(Parser.configs.get('CellDetection', '__do_mdrnn'))