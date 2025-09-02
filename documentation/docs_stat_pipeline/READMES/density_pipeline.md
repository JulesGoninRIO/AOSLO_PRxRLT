# Density Analysis Pipeline Documentation

## Overview

The **Density Analysis Pipeline** processes AOSLO (Adaptive Optics Scanning Laser Ophthalmoscopy) images to extract cone photoreceptor densities and correlate them with retinal layer thicknesses from OCT data. The pipeline applies Yellott's ring method for density extraction and performs statistical correlation analysis between density measurements and layer thickness values.

## Pipeline Entry Point

**File:** `density_analysis_pipeline_manager.py`  
**Main Class:** [`DensityAnalysisPipelineManager`](density_analysis_pipeline_manager.py)

### Pipeline Flow

The pipeline can be initiated through the `main()` function, which orchestrates the following sequence:

1. **Path Management Setup**
   - Creates a [`ProcessingPathManager`](processing_path_manager.py) instance to handle all file paths and directory structure
   - Manages paths to input data, output directories, and intermediate results

2. **Montage Loading/Building** 
   - Attempts to load pre-existing montage from pickle file (`mosaic.pkl`)
   - If pickle doesn't exist or is corrupted, builds montage from scratch using [`CorrectedMontageMosaicBuilder`](montage_mosaic_builder.py)
   - The montage reconstructs the retinal mosaic from individual AOSLO images, establishing a fovea-centered coordinate system

3. **Pipeline Execution**
   - Instantiates [`DensityAnalysisPipelineManager`](density_analysis_pipeline_manager.py) with the path manager, montage mosaic, and step size (0.1°)
   - Calls the `run()` method with specified parameters to execute the analysis

### Core Pipeline Stages

The `run()` method manages four key processing stages:

#### Stage 1: Cone Density Extraction
**Options:**
- **From CSV:** Loads pre-computed fitted densities from `densities.csv`
- **From Raw CSV:** Loads raw density measurements and performs fitting
- **From Scratch:** Computes densities directly from AOSLO images using [`ConeDensityCalculator`](cone_density_calculator.py)

**Process:**
- Uses [`ConeDensityCalculator`](cone_density_calculator.py) class to apply Yellott's ring method
- Extracts cone densities at different eccentricities from the fovea
- Fits bilateral theoretical model to correct systematic errors near fovea and periphery
- Creates [`Density`](density.py) object containing raw, smoothed, and fitted density data

#### Stage 2: Layer Thickness Processing (Optional)
**Enabled by:** `do_layer=True` parameter

**Process:**
- Uses [`LayerCollection`](layer_collection.py) class to load OCT-derived layer thickness data
- Applies [`LayerThicknessCalculator`](layer_thickness_calculator.py) to compute thicknesses at matching eccentricities
- Processes multiple retinal layers: RNFL, GCL+IPL, INL+OPL, ONL, PR+RPE, Choroid, OS
- Ensures spatial alignment with AOSLO coordinate system through fovea-centered registration

#### Stage 3: Density-Layer Correlation Analysis
**Process:**
- Creates [`DensityLayerAnalysis`](density_layer_analysis.py) instance to compare cone densities with layer thicknesses
- Performs correlation analysis using Spearman's coefficients for each meridian
- Generates visualization plots comparing densities and thicknesses
- Outputs results to `density_analysis_new` subdirectory

#### Stage 4: Results Output and Debugging
**Outputs:**
- Statistical correlation results (`spearman_*.txt`)
- Summary data combining densities and thicknesses (`results.csv`)
- Visualization plots (`*_thickness_curves.png`)
- Debug thickness curve plots for quality control

### Execution Modes

#### Single Subject Processing
Processes individual subject/session by specifying the path to their data directory:
```
Subject###/Session###/
├── montaged/                    # MATLAB montaging outputs
├── montaged_corrected/         # Final corrected montage
├── density_analysis_new/       # Pipeline outputs
└── layer_new/                 # OCT layer data
```

#### Batch Processing
Iterates through the entire healthy subject cohort, processing all valid subject/session combinations found in the base directory structure.

### Key Parameters

- **`do_montage`**: Whether to load/build the montage mosaic (required for density calculation from scratch)
- **`do_layer`**: Enable layer thickness processing and correlation analysis
- **`from_csv`**: Load pre-fitted densities from CSV instead of calculating
- **`from_csv_raw`**: Load raw densities and perform fitting step
- **`to_csv_dens`**: Save computed densities to CSV files
- **`step`**: Eccentricity step size in degrees (default: 0.1°)

### Dependencies

The pipeline manager coordinates several modules:
- **Montaging:** [`MontageMosaic`](montage_mosaic.py), [`CorrectedMontageMosaicBuilder`](montage_mosaic_builder.py)
- **Density Analysis:** [`ConeDensityCalculator`](cone_density_calculator.py), [`Density`](density.py)
- **Layer Processing:** [`LayerCollection`](layer_collection.py), [`LayerThicknessCalculator`](layer_thickness_calculator.py)
- **Correlation Analysis:** [`DensityLayerAnalysis`](density_layer_analysis.py)
- **Path Management:** [`ProcessingPathManager`](processing_path_manager.py)

### Error Handling

The pipeline includes error handling for batch processing, continuing with remaining subjects if individual processing fails. Debug information and timing metrics are provided throughout execution to monitor performance and identify bottlenecks.