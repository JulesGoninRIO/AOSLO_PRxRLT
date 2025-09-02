# Helper Classes Documentation

## Overview

The pipeline uses several helper classes as data structures and utility components across different processing stages. These classes organize measurement data, manage coordinate systems, and handle spatial calculations required by the analysis components.

## Core Data Structures

### Density Class

**File:** [`density.py`](density.py)

The [`Density`](density.py) class serves as the primary data container for cone density measurements throughout the analysis pipeline. Implemented as a dataclass, it organizes density data across retinal meridians and processing stages.

#### Data Organization Structure

The class maintains separate dictionaries for horizontal (X) and vertical (Y) meridian measurements, with each meridian containing three processing stages: raw measurements from Yellott's method, smoothed data using LOWESS regression, and fitted data from bilateral theoretical models.

The coordinate system conventions follow anatomical orientation standards. Negative X values correspond to the temporal retinal region, while positive X values represent the nasal region. For the Y-axis, negative values indicate the superior retinal region, and positive values represent the inferior region. This coordinate system aligns with the fovea-centered analysis approach used throughout the pipeline.

#### Data Structure Implementation

Each density type (raw, smoothed, fitted) uses dictionary structures mapping eccentricity values in degrees to density measurements in cones per square millimeter. The dictionary approach provides efficient lookup operations while maintaining the sparse data structure necessary for handling missing measurements at certain eccentricities.

The fitted data represents the output of bilateral theoretical model optimization, providing physiologically constrained density estimates across the full eccentricity range. The smoothed data applies LOWESS regression to reduce measurement noise while preserving spatial trends. Raw data contains the direct output from Yellott's ring analysis before any post-processing.

#### Integration with Pipeline Components

The Density class interfaces with multiple pipeline components through standardized dictionary access patterns. The [`ConeDensityCalculator`](cone_density_calculator.py) populates the raw and fitted measurements, while visualization components access the data for plotting density curves and correlation analysis.

The class design supports both legacy single-direction processing and modern bilateral analysis approaches through the deprecated `add_density` method and the current dictionary-based access patterns.

### Direction Enumeration

**File:** [`direction.py`](direction.py)

The Direction enumeration provides type-safe specification of retinal meridians for analysis operations. This enumeration standardizes meridian references across different pipeline components and prevents coordinate system confusion.

The enumeration defines X and Y direction constants with associated properties for meridian identification. The `is_X` and `is_Y` properties enable conditional processing logic based on meridian orientation, supporting the bilateral analysis approach used in density fitting and layer correlation analysis.

## Montage System Classes

### MontageElement Class

**File:** [`montage_element.py`](montage_element.py)

The [`MontageElement`](montage_element.py) class implements the Composite pattern to combine image data and spatial transformation information. This design keeps the [`ImageFile`](image_file.py) class lightweight while giving [`MontageElement`](montage_element.py) spatial analysis functionality.

#### Transformation Management

The class maintains multiple transformation representations to handle different coordinate system requirements. The `fundus_transform` represents the image position relative to the fundus coordinate system established during GUI correction. The `transform` attribute contains the final transformation matrix applied during montage construction, incorporating scaling factors and coordinate system corrections.

Transformation application occurs through the `set_transform` method, which applies scaling ratios to convert between fundus coordinates and final montage coordinates. This dual-transformation approach enables consistent spatial relationships while accommodating different analysis requirements.

#### Spatial Analysis Integration

The class supports spatial analysis through region of interest (ROI) management for both ATMS and MDRNN cone detection algorithms. The `roi_atms` and `roi_mdrnn` lists contain [`RegionOfInterest`](montage_element.py) objects that define analysis regions within the image bounds.

Neighbor identification through the `build_neighbors` method enables overlap analysis and spatial consistency checks across adjacent images. The neighbor calculation uses the logical grid positions from image acquisition rather than actual spatial distances, reflecting the systematic sampling pattern of AOSLO imaging.

#### Image Transformation Operations

The `get_transformed_data` method applies affine transformations to image data using OpenCV's `warpAffine` function. The method calculates transformation boundaries through corner point transformation and creates output images sized to contain the complete transformed result.

Caching mechanisms prevent redundant transformation calculations during repeated access operations. The transformed data remains available through the `transformed_data` attribute until explicitly cleared or recalculated.

### MontageMosaic Class

**File:** [`montage_mosaic.py`](montage_mosaic.py)

The [`MontageMosaic`](montage_mosaic.py) class manages the complete retinal mosaic construction and establishes the coordinate system for subsequent analysis operations. The class handles element collection, coordinate system setup, and mosaic generation with caching for performance.

#### Coordinate System Management

The mosaic establishes a fovea-centered coordinate system through the `center_point` and `transformed_center` attributes. The center point calculation depends on external fovea identification, typically from OCT landmark analysis or manual specification during montage correction.

Coordinate system scaling occurs through the `ratio` parameter, which accounts for differences between fundus image resolution and final analysis resolution. The scaling application maintains spatial relationships while optimizing memory usage and processing performance.

#### Mosaic Construction Pipeline

The mosaic construction process operates through several coordinated methods. The `_prepare_mosaic` method establishes the output array dimensions based on transformed element boundaries or pre-specified dimensions. Element transformation and positioning occur through the `_create_mosaic` method, which applies individual element transformations and combines results through pixel accumulation.

Boundary calculation through the `get_corners` helper function determines the minimum bounding rectangle containing all transformed elements. The `correct_transforms` function adjusts element transformations to align with the calculated boundaries, establishing consistent coordinate origins.

#### Caching and Performance Optimization

The class implements comprehensive caching for both the final mosaic image (`_mosaic_cache`) and binary analysis maps (`_binary_map_cache`). Cache invalidation occurs when transformation parameters change or new elements are added to the mosaic.

Binary map generation creates boolean masks indicating valid analysis regions by excluding vessel shadows and image boundaries. The binary map generation uses the same transformation pipeline as the primary mosaic but applies different combination logic to maintain boolean output.

#### Analysis Integration Support

The mosaic creates spatial analysis support through overlap region calculation and neighbor identification methods. The `get_overlap_between_neighbors` method calculates intersection regions between adjacent images for overlap analysis and spatial consistency validation.

Element retrieval methods give analysis components access to individual images within the mosaic context. The `get_element` and `get_transform` methods offer consistent access patterns for retrieving elements by name or [`ImageFile`](image_file.py) reference.

### RegionOfInterest Classes

**File:** [`montage_element.py`](montage_element.py)

The RegionOfInterest class hierarchy defines spatial analysis regions for cone detection and density analysis operations. The base [`RegionOfInterest`](montage_element.py) class defines the interface, while specific implementations handle different geometric shapes and analysis requirements.

#### DiskROI Implementation

The [`DiskROI`](montage_element.py) class represents circular analysis regions defined by center coordinates and radius parameters. This implementation supports the circular region analysis used in Yellott's ring method and traditional cone counting approaches.

The disk transformation through the `transform` method applies affine transformations to the center coordinates while maintaining the radius in the original coordinate system. This approach preserves the circular shape characteristics necessary for consistent analysis region definition.

Visualization through the `draw` method enables region overlay on images for quality control and analysis validation. The drawing implementation uses OpenCV circle drawing functions with configurable color parameters for different analysis contexts.

#### Integration with Analysis Pipeline

Region of interest objects integrate with analysis pipelines through the density calculation and cell detection components. The spatial analysis components iterate through ROI collections to extract measurements at defined locations and scales.

The reference point attribute provides coordinate information for eccentricity calculations and spatial correlation analysis. The optional density attribute enables result storage directly within the ROI object for simplified data management.

## Implementation Considerations

### Memory Management

The helper classes implement memory management strategies for large-scale image analysis operations. The [`MontageMosaic`](montage_mosaic.py) caching system prevents redundant calculations while managing memory usage through lazy evaluation and cache invalidation.

Dictionary-based data structures in the [`Density`](density.py) class provide memory-efficient storage for sparse measurement data. The sparse structure accommodates missing measurements and irregular sampling patterns without allocating unused storage.

### Coordinate System Consistency

All helper classes maintain consistent coordinate system conventions aligned with the anatomical orientation standards. The transformation management in [`MontageElement`](montage_element.py) and [`MontageMosaic`](montage_mosaic.py) preserves spatial relationships across different processing stages and analysis requirements.

Error handling in coordinate transformations prevents invalid spatial calculations and provides diagnostic information for troubleshooting coordinate system misalignment issues.

### Performance Optimization

The helper classes use performance optimization including transformation caching, lazy evaluation, and efficient data structure selection. The composite pattern implementation in [`MontageElement`](montage_element.py) separates lightweight data containers from computationally intensive spatial operations.

Vectorized operations in transformation calculations utilize NumPy broadcasting and OpenCV optimized functions for maximum performance on large image datasets.