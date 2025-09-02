# MDRNN Pipeline Detailed Documentation

## Table of Contents

- [Overview](#overview)
  - [Input and Output](#input-and-output)
- [MDRNNPipelineManager](#mdrnnpipelinemanager)
  - [Purpose](#purpose)
  - [Initialization](#initialization)
  - [Core Methods](#core-methods)
    - [`run()`](#run)
    - [`__run_best_method()`](#__run_best_method)
    - [`_run(mdrnn_preprocessing_params)`](#_runmdrnn_preprocessing_params)
  - [Configuration Dependencies](#configuration-dependencies)
- [MDRNNAlgorithm](#mdrnalgorithm)
  - [Purpose](#purpose-1)
  - [Class Constants](#class-constants)
  - [Core Methods](#core-methods-1)
    - [`run(image_ids: List[str])`](#runimage_ids-liststr)
    - [`__generate_lut_file()`](#__generate_lut_file)
    - [`__run_mdrnn()`](#__run_mdrnn)
    - [`reorder_results()`](#reorder_results)
  - [External Algorithm Integration](#external-algorithm-integration)
    - [`run_in_conda_env(env_name, data_folder, lut_csv)`](#run_in_conda_envenv_name-data_folder-lut_csv)
- [MDRNNPostProcessor](#mdrnnpostprocessor)
  - [Purpose](#purpose-2)
  - [Initialization](#initialization-1)
  - [Core Methods](#core-methods-2)
    - [`gather_statistics()`](#gather_statistics)
    - [`save_run(new_name: str)`](#save_runnew_name-str)
    - [`get_statistics(number_cones_detected_per_patch)`](#get_statisticsnumber_cones_detected_per_patch)
  - [Helper Functions](#helper-functions)
    - [`get_separated_patches(patch_dir, cones_detected, threshold)`](#get_separated_patchespatch_dir-cones_detected-threshold)
- [MDRNNPreProcessor](#mdrnnpreprocessor)
  - [Purpose](#purpose-3)
  - [Initialization](#initialization-2)
  - [Core Pipeline Methods](#core-pipeline-methods)
    - [`run() -> List[ImageFile]`](#run---listimageFile)
    - [`crop_images_to_patches() -> List[ImageFile]`](#crop_images_to_patches---listimageFile)
    - [`apply_preprocessing_to_images(all_patches)`](#apply_preprocessing_to_imagesall_patches)
  - [Preprocessing Techniques](#preprocessing-techniques)
    - [`preprocess_patch(patch, corr_patch)`](#preprocess_patchpatch-corr_patch)
    - [Neural Style Transfer Integration](#neural-style-transfer-integration)
  - [Statistical Processing Methods](#statistical-processing-methods)
    - [Range-Based Corrections](#range-based-corrections)
  - [Histogram Processing](#histogram-processing)
    - [`compute_curring_histogram() -> Tuple[float, float, float]`](#compute_curring_histogram---tuplefloat-float-float)
    - [`pixel_histogram_per_patch(patch_dir) -> Dict[ImageFile, np.array]`](#pixel_histogram_per_patchpatch_dir---dictimageFile-nparray)
- [Integration and Data Flow](#integration-and-data-flow)
  - [Pipeline Coordination](#pipeline-coordination)
  - [Data Dependencies](#data-dependencies)
  - [File System Organization](#file-system-organization)
- [Dependencies](#dependencies)
  - [System Requirements](#system-requirements)
  - [Pipeline Dependencies](#pipeline-dependencies)

## Overview

This document provides detailed technical documentation for the core components of the NST/MDRNN cell detection pipeline. The pipeline consists of four interconnected classes that work together to transform raw AOSLO retinal images into quantified cone photoreceptor detection results through a machine learning workflow.

The architecture centers around the `MDRNNPipelineManager`, which serves as the central coordinator orchestrating the entire detection process. This manager determines the appropriate preprocessing strategy based on configuration parameters, coordinates the execution sequence, and ensures proper data flow between components. The manager makes decisions about whether to apply Neural Style Transfer enhancement or use conventional preprocessing techniques, determining how the subsequent analysis will proceed.

The preprocessing phase is handled by `MDRNNPreProcessor`, which transforms full retinal images into analysis-ready patches through a multi-stage enhancement process. This component performs the task of converting raw AOSLO images into standardized patches while applying various enhancement techniques including histogram matching, statistical outlier correction, and optionally Neural Style Transfer. The preprocessor handles image segmentation, manages zero-pixel replacement strategies, and can apply enhancement algorithms that modify the input data for the machine learning algorithm.

Once preprocessing is complete, the `MDRNNAlgorithm` component manages the interface with the external Multi-Directional Recurrent Neural Network cone detection algorithm. This component handles the execution of a TensorFlow-based machine learning model within a separate conda environment, managing file system organization for algorithm inputs and outputs, and coordinating the external process execution. The algorithm component creates lookup tables that specify which patches to analyze, manages the execution of Davidson's MDRNN cone detector, and handles the reorganization of raw algorithm outputs into structured coordinate data.

The final phase involves `MDRNNPostProcessor`, which analyzes the detection results to generate quality metrics and statistical summaries. This component applies Otsu thresholding to categorize patches based on detection success, generates visualization plots for quality assessment, and produces structured statistical analyses that can be used to evaluate the effectiveness of different preprocessing approaches. The postprocessor also handles the organization and archival of completed processing runs, ensuring that results from different preprocessing methods can be compared and analyzed.

### Input and Output

**Pipeline Inputs:**
- **`montage_mosaic`**: MontageMosaic object with spatially reconstructed AOSLO images from montaging pipeline, fovea-centered coordinate system
- **`path_manager`**: ProcessingPathManager instance managing subject/session file organization and directory structure
- **`preprocessing_params`**: MDRNNPreProcessingParams instance determining enhancement methods (NST vs conventional preprocessing)
- **`ImageModalities.CS.value`**: Confocal scanning images containing cone photoreceptor structures, filtered during image processing

**Pipeline Outputs:**
- **`algorithmMarkers/`**: Directory containing CSV files with cone coordinates for each processed patch
- **`center.csv`**: Raw algorithm output file with detection results before reorganization
- **`preprocessed_patches`**: List of processed patch identifiers tracking preprocessing completion
- **`separated_patches`**: Dictionary with `"good"` and `"bad"` keys categorizing patches based on detection success
- **`raw_run/`**: Directory (`path_manager.mdrnn.raw_run_path`) storing preprocessed image patches for debugging
- **`{preprocessing_method}/`**: Method-specific result directories named using `str(preprocessing_params)`
- **Analysis plots**: Otsu threshold plots, preprocessing comparison charts, and detection quality visualizations
- **Algorithm logs**: Execution stdout/stderr from `run_in_conda_env()` for troubleshooting

---

## MDRNNPipelineManager

**File:** `src/cell/cell_detection/mdrnn/mdrnn_pipeline_manager.py`  
**Class:** `MDRNNPipelineManager`

### Purpose
Central orchestrator for the MDRNN pipeline, coordinating preprocessing, algorithm execution, and postprocessing steps.

### Initialization
```python
def __init__(self, path_manager: ProcessingPathManager, montage_mosaic):
```

**Parameters:**
- `path_manager`: ProcessingPathManager instance for file system management
- `montage_mosaic`: MontageMosaic instance containing spatial image data

**Initialization Process:**
- Initializes MDRNN-specific paths via `path_manager.mdrnn.initialize_mdrnn()`
- Creates `MDRNNPostProcessor` and `MDRNNAlgorithm` instances
- Stores montage mosaic reference for spatial context

### Core Methods

#### `run()`
Main execution method that initiates the MDRNN pipeline using optimal parameters.
- Calls `__run_best_method()` internally
- Serves as the primary entry point for pipeline execution

#### `__run_best_method()`
Determines and executes the appropriate preprocessing method based on configuration.

**Execution Logic:**
```python
if self.path_manager.mdrnn.run_nst:
    preprocessing_methods = [build_parameters_best_method()]
else:
    preprocessing_methods = [build_parameters_best_preprocessing()]
```

**Method Selection:**
- **NST Mode**: Uses `build_parameters_best_method()` for Neural Style Transfer preprocessing
- **Standard Mode**: Uses `build_parameters_best_preprocessing()` for conventional preprocessing

#### `_run(mdrnn_preprocessing_params)`
Executes the complete pipeline sequence for a given preprocessing configuration.

**Process Flow:**
1. **Path Cleanup**: `self.path_manager.mdrnn.clean_mdrnn()`
2. **Preprocessing**: Creates and runs `MDRNNPreProcessor`
3. **Algorithm Execution**: Runs `MDRNNAlgorithm` on preprocessed patches
4. **Postprocessing**: Gathers statistics and saves results with specified naming

**Naming Convention:**
- Uses `str(mdrnn_preprocessing_params)` for custom preprocessing
- Falls back to `path_manager.mdrnn.first_run_dir` for default processing

### Configuration Dependencies
- Relies on `Parser` configuration for NST mode determination
- Uses `build_parameters_best_preprocessing()` and `build_parameters_best_method()` from `mdrnn_constants`

---

## MDRNNAlgorithm

**File:** `src/cell/cell_detection/mdrnn/mdrnn_algorithm.py`  
**Class:** `MDRNNAlgorithm`

### Purpose
Interfaces with the external MDRNN cone detection algorithm, managing file preparation, algorithm execution, and result organization.

### Class Constants
```python
__lut_filename = MDRNN_LUT_FILENAME
__cone_detector_prepr_dir = MDRNN_CONE_DETECTOR_PREP_DIR
__cone_detector_out_dir = MDRNN_CONE_DETECTOR_OUT_DIR
__python_env = PYTHON_ENV
```

### Core Methods

#### `run(image_ids: List[str])`
Main execution method that orchestrates the complete algorithm workflow.

**Process Sequence:**
1. **LUT Generation**: Creates lookup table file for algorithm input
2. **Algorithm Execution**: Runs MDRNN algorithm via `__run_mdrnn()`
3. **Directory Management**: Renames output directory to `MDRNN_CONE_DETECTOR_OUT_DIR`
4. **File Organization**: 
   - Moves CSV and pickle files to output directory
   - Copies TIFF files to `_raw` directory
5. **Result Processing**: Calls `reorder_results()` for output organization

#### `__generate_lut_file()`
Creates CSV lookup table file containing patch locations and confidence thresholds.

**File Format:**
```
CROP_{file_id},0.8
```
- Each line represents one patch with 0.8 confidence threshold
- Written to `output_path / lut_filename`

#### `__run_mdrnn()`
Executes the MDRNN algorithm in a dedicated conda environment.

**Execution Process:**
- Calls `run_in_conda_env()` with specified Python environment
- Implements retry logic on failure with directory cleanup
- Logs warnings and errors from algorithm execution

**Error Handling:**
```python
if stderr:
    logging.warning(stderr)
    self.__remove_last_run_directory()
    # Retry execution
```

#### `reorder_results()`
Reorganizes algorithm output into structured format.

**Process:**
1. Reads `center.csv` containing detection results
2. Parses center coordinates using `preprocess_centers()`
3. Creates individual CSV files for each image in `algorithmMarkers/` directory
4. Outputs coordinate pairs as (x,y) format without headers

### External Algorithm Integration

#### `run_in_conda_env(env_name, data_folder, lut_csv)`
Executes MDRNN algorithm in isolated conda environment.

**Environment Configuration:**
- **Conda Path**: `C:\Users\BardetJ\AppData\Local\anaconda3\Scripts\conda.exe`
- **Environment Name**: `'tf'` (TensorFlow environment)
- **Model**: `'paperModel'`

**Command Structure:**
```python
arguments = ["--input_folder", str(data_folder)]
activate_env = f'"{conda_path}" run -n {env_name} python {current_path}\\old_cone_detector\\cone_detector\\davidson_cone_detector.py {" ".join(arguments)}'
```

**Output Processing:**
- Captures stdout and stderr using temporary files
- Filters stderr for actual errors vs. warnings
- Returns processed output streams

---

## MDRNNPostProcessor

**File:** `src/cell/cell_detection/mdrnn/mdrnn_postprocessor.py`  
**Class:** `MDRNNPostProcessor`

### Purpose
Handles post-processing analysis of MDRNN results, including statistical analysis and visualization generation.

### Initialization
```python
def __init__(self, path_manager: ProcessingPathManager, previous_path: Path = None):
```

### Core Methods

#### `gather_statistics()`
Analyzes MDRNN detection results and generates statistical summaries.

**Process:**
1. **Patch Analysis**: Calls `pixels_per_patch()` on raw run directory
2. **Statistics Generation**: Processes cone detection counts per patch
3. **Separation Analysis**: Categorizes patches based on detection success

#### `save_run(new_name: str)`
Renames and organizes the completed processing run.

**Functionality:**
- Renames `postprocessed` folder to specified method name
- Handles existing directory conflicts with cleanup
- Implements retry logic for file system operations

#### `get_statistics(number_cones_detected_per_patch)`
Comprehensive statistical analysis of detection results.

**Analysis Pipeline:**
1. **Cone Counting**: `get_number_of_cones_per_patch()` from algorithm markers
2. **Histogram Generation**: `create_histogram_distribution()` of detection counts
3. **Threshold Calculation**: `get_otsu_threshold()` for patch quality classification
4. **Visualization**: Creates analysis plots via `CellDetectionPlotter`
5. **Patch Separation**: `get_separated_patches()` into "good" and "bad" categories

**Output Generation:**
- Creates analysis directory if specified
- Generates Otsu threshold plots
- Produces preprocessing analysis plots

### Helper Functions

#### `get_separated_patches(patch_dir, cones_detected, threshold)`
Categorizes patches based on detection quality.

**Classification Logic:**
```python
separated_patches = {"good": [], "bad": []}
if cones_detected[patch] > threshold:
    separated_patches["good"].append(patch)
else:
    separated_patches["bad"].append(patch)
```

**Output:** Dictionary with "good" and "bad" patch lists based on threshold comparison

---

## MDRNNPreProcessor

**File:** `src/cell/cell_detection/mdrnn/mdrnn_preprocessor.py`  
**Class:** `MDRNNPreProcessor`

### Purpose
Handles comprehensive image preprocessing including patch extraction, enhancement techniques, and Neural Style Transfer preparation.

### Initialization
```python
def __init__(self, path_manager: ProcessingPathManager, montage_mosaic: MontageMosaic = None, preprocessing_params: MDRNNPreProcessingParams = None):
```

**Configuration:**
- Computes `curring_histogram` when preprocessing parameters are provided
- Initializes empty `preprocessed_patches` list for tracking

### Core Pipeline Methods

#### `run() -> List[ImageFile]`
Main preprocessing execution method.

**Process Flow:**
1. **Patch Extraction**: `crop_images_to_patches()`
2. **Enhancement Application**: `apply_preprocessing_to_images()`
3. **Return Results**: List of preprocessed patch identifiers

#### `crop_images_to_patches() -> List[ImageFile]`
Converts full images into analysis patches.

**Process:**
1. **Image Filtering**: Processes only Confocal Scanning (`CS`) modality images
2. **Image Loading**: Creates `ImageFile` instances with data reading
3. **Zero Replacement**: Applies `preprocess_images()` for zero-pixel handling
4. **Patch Generation**: 
   - Direct processing for pre-cropped images (with prefix)
   - `PatchCropper` usage for full-size images

#### `apply_preprocessing_to_images(all_patches)`
Applies specified preprocessing techniques to patch collection.

**Enhancement Mode Configuration:**
```python
if self.preprocessing_params.enhancement == "match" or self.preprocessing_params.enhancement == "nst":
    # Load previous processing results for comparison
    previous_name = str(self.preprocessing_params)
    previous_dir = re.sub("_" + self.preprocessing_params.enhancement, "", previous_name)
    histogram_per_patch = pixel_histogram_per_patch(self.previous_path)
    corr_patch = CorrespondingPatchFinder(...).get_corresponding_patches()
```

### Preprocessing Techniques

#### `preprocess_patch(patch, corr_patch)`
Applies comprehensive preprocessing to individual patches.

**Processing Options:**
1. **Range Processing**: Lower, upper, or both range corrections
2. **Zero Method**: Special handling for zero-value pixels
3. **Histogram Matching**: Statistical distribution alignment
4. **Neural Style Transfer**: Style-based enhancement

**Method Selection Logic:**
```python
if self.preprocessing_params.method:
    if self.preprocessing_params.range_method == "lower":
        self.process_lower_range(patch.data)
    elif self.preprocessing_params.range_method == "both":
        self.process_both_ranges(patch.data)
    else:
        self.process_zero_method(patch.data)
```

#### Neural Style Transfer Integration

**NST Application Process:**
```python
def apply_nst_to_patches(self, patch: ImageFile, corr_patch: Dict[ImageFile, ImageFile]):
    if patch in corr_patch:
        style_image_file = corr_patch[patch]
        nst_pipeline_manager = NSTPipelineManager(style_image_file, patch)
        cropped = nst_pipeline_manager.run()
        nst_pipeline_manager.generate_plot(self.path_manager.mdrnn.output_path)
        nst_pipeline_manager.save_losses(self.path_manager.mdrnn.output_path)
        save_image(cropped, str(patch_name))
```

### Statistical Processing Methods

#### Range-Based Corrections

**Median Replacement (`process_median_replace`)**:
- **Local Correction**: Uses patch-specific statistics (mean, variance/std)
- **Global Correction**: Uses dataset-wide `curring_hist` statistics
- **Range Options**: Lower, upper, or zero-value replacement

**Border Replacement (`process_border_replace`)**:
- Similar logic to median replacement
- Replaces values with computed boundary values instead of median

**Threshold Calculations:**
```python
# Variance-based thresholds
lower_threshold = np.mean(patch_data) - np.var(patch_data)
upper_threshold = np.mean(patch_data) + np.var(patch_data)

# Standard deviation-based thresholds  
lower_threshold = np.mean(patch_data) - np.std(patch_data)
upper_threshold = np.mean(patch_data) + np.std(patch_data)
```

### Histogram Processing

#### `compute_curring_histogram() -> Tuple[float, float, float]`
Calculates dataset-wide statistical parameters for global corrections.

**Returns:** (mean, variance/std, median) tuple based on method parameter

#### `pixel_histogram_per_patch(patch_dir) -> Dict[ImageFile, np.array]`
Generates histogram distributions for each patch in specified directory.

**Process:**
- Reads all TIFF files in patch directory
- Computes 256-bin histograms for pixel value range [0, 256]
- Returns mapping of ImageFile to histogram array

---

## Integration and Data Flow

### Pipeline Coordination
1. **MDRNNPipelineManager** orchestrates the complete workflow
2. **MDRNNPreProcessor** handles image preparation and enhancement
3. **MDRNNAlgorithm** executes external cone detection algorithm
4. **MDRNNPostProcessor** analyzes results and generates statistics

### Data Dependencies
- **Input**: AOSLO confocal images from montage mosaic
- **Intermediate**: Preprocessed patches, LUT files, algorithm results
- **Output**: Cone detection coordinates, analysis statistics, quality plots

### File System Organization
```
Subject###/Session###/
├── postprocessed_atms_single/       # Final MDRNN results
├── raw_run/                         # Intermediate patch files
├── {preprocessing_method}/          # Method-specific results
└── checkpoint/                      # Algorithm checkpoints
```

---

## Dependencies

### System Requirements
- **Python Environment**: Conda environment 'tf' with TensorFlow
- **External Algorithm**: Davidson's MDRNN cone detector in `old_cone_detector/`
- **Image Processing**: OpenCV, NumPy for image manipulation
- **Visualization**: Matplotlib for plotting and analysis

### Pipeline Dependencies
- **[ProcessingPathManager](processing_path_manager.md)**: File system organization
- **[MontageMosaic](montage_mosaic.md)**: Spatial image data structure
- **[NSTPipelineManager](nst_pipeline_manager.md)**: Neural Style Transfer processing
- **[PatchCropper](patch_cropper.md)**: Image segmentation into patches
- **[CorrespondingPatchFinder](corresponding_patch_finder.md)**: Patch matching for enhancement

