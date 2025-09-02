from src.cell.cell_detection.mdrnn.mdrnn_preprocessing_params import MDRNNPreProcessingParams

OUTPUT_DIR = 'postprocessed'
MDRNN_CONE_DETECTOR_PREP_DIR = '_prepocessing'
MDRNN_CONE_DETECTOR_OUT_DIR = 'output_cone_detector'
MDRNN_LUT_FILENAME = "lut.csv"
ANALYSIS_DIR = "_analysis"
PLOT_DIR = "plot"
PYTHON_ENV = "python37"

FIRST_RUN_DIR_NAME = "raw_run"
RAW_DATA_DIR_NAME = "_raw"

# best method parameters
BEST_METHOD = "var" # for variance
BEST_REPLACE = "border"
BEST_RANGE_METHOD = "both"
BEST_ENHANCEMENT = "nst"
BEST_CORRECT = "global"

# whether you want to run the algorithm on raw data (no preprocessing)
RAW_RUN = True

# whether you want to optimize for the best preprocesing method
DO_ANALYSIS = False

# solo = True -> multiple NST is not implemented in this version
VISUALIZE_PATCHES = False

def build_parameters_best_method() -> str:
    """
    Static method to initialize the best method

    :return: dict with the best method
    """

    mdrnn_preprocessing_params = build_parameters_best_preprocessing()
    mdrnn_preprocessing_params.enhancement = BEST_ENHANCEMENT

    return mdrnn_preprocessing_params

def build_parameters_best_preprocessing() -> str:
    """
    Static method to initialize the best preprocessing method

    :return: dict with the best preprocessing method
    """

    method = BEST_METHOD
    replace = BEST_REPLACE
    range_method = BEST_RANGE_METHOD
    enhancement = None
    correct = BEST_CORRECT

    mdrnn_preprocessing_params = MDRNNPreProcessingParams(
            method=method,
            replace=replace,
            range_method=range_method,
            enhancement=enhancement,
            correct=correct
        )

    return mdrnn_preprocessing_params
