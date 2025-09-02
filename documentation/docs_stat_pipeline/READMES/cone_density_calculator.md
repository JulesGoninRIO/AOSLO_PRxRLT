# Montage Mosaic Builder Documentation

## Overview

The Montage Mosaic Builder module constructs retinal mosaics from individual AOSLO images by applying geometric transformations to align and position images within a fovea-centered coordinate system. The module implements multiple builder strategies to handle different input data sources and correction states.

## Architecture

**File:** `montage_mosaic_builder.py`  
**Base Class:** [`MontageMosaicBuilder`](montage_mosaic_builder.py) (Abstract)

The module follows the Builder pattern with four concrete implementations, each handling different scenarios for montage construction. The base class defines the common interface and shared functionality, while concrete builders implement specific transformation and alignment strategies.

### Builder Classes

#### MontageMosaicBuilder (Abstract Base)
The abstract base class provides common functionality for element management and montage construction. It manages a collection of [`MontageElement`](montage_element.py) objects and coordinates with the [`MontageMosaic`](montage_mosaic.py) class to build the final mosaic structure.

**Core Operations:**
- Element addition with optional transforms
- Corner calculation and transform correction
- Center point adjustment based on montage boundaries

#### MatlabMontageMosaicBuilder
Processes transformation data directly from MATLAB montaging outputs. This builder reads `.mat` files containing transformation matrices and applies inverse transformations to reconstruct the montage coordinate system.

**Process Flow:**
- Reads transformation matrices from [`MatlabReader`](matlab_reader.py)
- Applies inverse perspective transformations using pseudo-inverse calculations
- Combines with global offset transformations to establish montage coordinates
- Handles the mathematical transformations inherited from the MATLAB montaging algorithm

#### CorrectedMontageMosaicBuilder (Primary)
The main builder used in production, which processes manually corrected montage data from the GUI correction step. This builder handles the most complex transformation pipeline, combining scaling, rotation, translation, and perspective corrections.

**Input Processing:**
The builder reads corrected positioning data from `locations.csv`, which contains component positions manually adjusted through the montaging GUI. The CSV structure includes center point coordinates, component locations, rotation angles, and image dimensions.

**Transformation Pipeline:**
The builder applies a multi-stage transformation to each image element. First, it loads the original image aligned to its reference component using the naming convention `{image}_aligned_to_ref{component}_m1.tif`. The image undergoes scaling transformation to match the required dimensions, followed by rotation transformation applied around the image center using OpenCV's rotation matrix calculation.

Translation transformation positions the scaled and rotated image at its corrected location within the fundus coordinate system. The builder then applies perspective transformation to map the image from its local coordinate system to the global montage space, calculating the perspective transform matrix from corner points of the transformed image.

**Coordinate System Management:**
The builder maintains alignment between the AOSLO montage coordinate system and the fundus image coordinate system. It calculates the center point from the first entry in the corrections file and establishes the ratio between image dimensions and fundus dimensions for proper scaling.

#### RawImageMosaicBuilder
A fallback builder for cases where MATLAB `.mat` files are unavailable. This builder attempts to extract transformation data directly from montage images, though it requires additional implementation for complete functionality.

### Key Processing Methods and Implementation Details

### Key Processing Methods and Implementation Details

#### Geometric Transformations
The transformation pipeline complexity stems from coordinate system conversions between MATLAB, OpenCV, and the GUI correction coordinate spaces. MATLAB's transformation matrices require conversion through pseudo-inverse calculations to work with Python's coordinate conventions.

Image scaling preserves relationships between GUI-corrected positions and full-resolution coordinates. Users position downscaled components in the montaging GUI for performance, requiring scaling transformations to map back to actual image dimensions.

Rotation transformations for non-zero angles require bounding box recalculation since rotation changes image dimensions. The `get_combined_transformation_matrix` method applies trigonometric calculations to prevent data loss during rotation.

Translation operates in fundus coordinates rather than image coordinates. The translation matrix divides fundus coordinates by scaling factors to account for coordinate system differences between the GUI workspace and final montage.

Perspective transformation maps from 720x720 pixel space to the arbitrary quadrilateral shape occupied in the montage after rotation and positioning.

#### Common Failure Modes and Debugging
**Missing Reference Components:** When `locations.csv` references components not found in the MATLAB output, the builder logs errors and continues. Check the `matched_chains` dictionary in `MatlabReader` for missing component mappings.

**Transform Matrix Errors:** Zero transforms indicate coordinate calculation failures. This typically occurs when GUI corrections place components at invalid positions or when rotation angles cause dimension overflow.

**Image Loading Failures:** The naming convention `{image}_aligned_to_ref{component}_m1.tif` assumes MATLAB preprocessing completed successfully. Missing files indicate incomplete MATLAB montaging.

**Memory Issues:** Large montages with many overlapping images can exceed memory limits during the `big_image` accumulation step. The padding operations in `pad_image_to_fundus_shape` create large arrays that may need optimization for datasets with extensive coverage.

#### Data Flow Dependencies
The builder requires specific input file structure from prior pipeline steps. MATLAB montaging must produce `.mat` files with transformation data and aligned image files. The GUI correction step must generate `locations.csv` with properly formatted coordinate and angle data.

The `center_point` calculation depends on the first entry in `locations.csv` being the fovea coordinates. Incorrect center point calculation propagates through all subsequent transformations, making manual verification essential.

#### Performance Considerations
The `CorrectedMontageMosaicBuilder` processes images sequentially, making it memory-intensive for large datasets. The `big_image` accumulation serves debugging purposes but consumes significant memory. For production use, this debug image generation could be disabled, but i have not checked yet.

SSIM correction (when enabled) significantly increases processing time as it searches neighborhoods around each component position. The search radius directly impacts both accuracy and computational cost.


### Integration with Pipeline

The Montage Mosaic Builder integrates with several other pipeline components. It receives path management through [`ProcessingPathManager`](processing_path_manager.py) and creates [`MontageMosaic`](montage_mosaic.py) instances populated with [`MontageElement`](montage_element.py) objects. The builder coordinates with [`MatlabReader`](matlab_reader.py) for transformation data and may utilize [`SSIMProcessing`](ssim_processing.py) for refinement corrections.

### Output Generation

The completed montage provides a unified coordinate system centered at the fovea, enabling subsequent analysis steps to work with consistent spatial references. The builder generates both the montage structure and the necessary transformation data for mapping between individual image coordinates and the global retinal coordinate system.

The final montage includes all successfully positioned image elements with their associated transformation matrices, center point location, and scaling factors needed for analysis pipeline integration. This output serves as the foundation for cone density extraction and layer thickness correlation analysis.