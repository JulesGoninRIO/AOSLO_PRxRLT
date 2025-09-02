# NST/MDRNN Cell Detection Pipeline Documentation

## Overview

The **NST/MDRNN Cell Detection Pipeline** processes AOSLO (Adaptive Optics Scanning Laser Ophthalmoscopy) images to detect and analyze photoreceptor cells using a Multi-Directional Recurrent Neural Network (MDRNN) approach combined with Neural Style Transfer (NST) preprocessing techniques. This pipeline performs automated cell detection and analysis.

## Pipeline Entry Points

### Primary Entry Point
**File:** `Start_PostProc_Pipe.py`  
**Main Function:** `main(df_to_process: pd.DataFrame = None)`

The main entry script serves as the unified entry point for all AOSLO postprocessing pipelines, including the NST/MDRNN cell detection pipeline. It handles environment configuration, logging setup, and pipeline orchestration.

```python
def main(df_to_process: pd.DataFrame = None):
    # Environment and logging configuration
    environment = os.getenv('ENVIRONMENT', 'development')
    logging.basicConfig(level=logging.WARNING if environment == 'production' else logging.DEBUG)
    
    # Pipeline initialization and execution
    postprocessing_pipeline = PipelineRunner(df_to_process)
    postprocessing_pipeline.go()
```

### Pipeline Orchestrator
**File:** `src/pipeline_engine/pipeline_runner.py`  
**Main Class:** [`PipelineRunner`](pipeline_runner.py)

The `PipelineRunner` class serves as the central orchestrator for all pipeline components. It reads configuration parameters, manages processing flags, and coordinates the execution of different analysis modules including the MDRNN cell detection pipeline.

## NST/MDRNN Pipeline Flow

### 1. Pipeline Initialization

The pipeline begins with the `PipelineRunner` initialization, which:

- **Configuration Loading**: Uses [`Parser`](../configs/parser.py) to read pipeline configuration from `config.txt`
- **Processing Flags**: Determines which pipeline components to execute (montaging, registration, MDRNN, etc.)
- **Dataset Preparation**: Identifies subject directories to process using [`prepare_dataset`](../shared/helpers/os_helpers.py)
- **DataFrame Setup**: Creates processing tracking DataFrame with flags for each pipeline step

```python
def __init__(self, df_to_process: pd.DataFrame):
    Parser.initialize()
    self.__do_mdrnn = Parser.do_mdrnn_cell_detection()
    # ... other configuration flags
    
    if df_to_process is not None:
        self.dirs_to_process = df_to_process['name']
    else:
        self.dirs_to_process = prepare_dataset(self.__base_dir, self.__do_blood_flow)
```

### 2. Prerequisite Processing

Before MDRNN cell detection can run, several prerequisite steps must be completed:

#### **Montage Construction**
- **Purpose**: Reconstructs retinal mosaic from individual AOSLO images
- **Component**: [`MontagePipelineManager`](../cell/montage/montage_pipeline_manager.py)
- **Output**: Fovea-centered coordinate system and aligned image mosaic
- **Requirement**: Required for spatial context for cell detection

#### **Image Registration** (Optional)
- **Purpose**: Aligns images across different imaging sessions or modalities
- **Component**: [`ImageRegistration`](../cell/registration/image_registration.py)
- **Output**: Spatially registered image datasets
- **Requirement**: Increases cell detection accuracy across time points

### 3. MDRNN Cell Detection Execution

**Process Flow:**
```python
mdrnn_dir_to_process = [dir_to_process['name'] for _, dir_to_process in 
    self.df_to_process.iterrows() if dir_to_process.mdrnn]

for dir_to_process in mdrnn_dir_to_process:
    processing_path_manager = ProcessingPathManager(Path(dir_to_process))
    montage_mosaic = processing_path_manager.get_montage_mosaic()
    self.__CellDetector_MDRNN = CellDetector_MDRNN(processing_path_manager, montage_mosaic)
    self.__CellDetector_MDRNN.run()
```

#### **Step 1: Path Management Setup**
- **Component**: [`ProcessingPathManager`](../cell/processing_path_manager.py)
- **Purpose**: Manages file paths and directory structure for each subject
- **Functionality**: Provides access to input images, output directories, and intermediate results

#### **Step 2: Montage Loading**
- **Method**: `processing_path_manager.get_montage_mosaic()`
- **Returns**: [`MontageMosaic`](../cell/montage/montage_mosaic.py) instance
- **Content**: Reconstructed retinal mosaic with spatial coordinate system

#### **Step 3: MDRNN Pipeline Manager Initialization**  
- **Component**: [`MDRNNPipelineManager`](../cell/cell_detection/mdrnn/mdrnn_pipeline_manager.py) (imported as `CellDetector_MDRNN`)
- **Parameters**: 
  - `processing_path_manager`: File system management
  - `montage_mosaic`: Spatial image data and coordinate system
- **Purpose**: Coordinates NST preprocessing and MDRNN cell detection

#### **Step 4: Pipeline Execution**
- **Method**: `self.__CellDetector_MDRNN.run()`
- **Process**: Executes the complete NST/MDRNN processing pipeline
- **Output**: Cell detection results, confidence maps, and analysis plots

## Key Pipeline Components

### Configuration Management
**File:** `src/configs/parser.py`  
**Class:** [`Parser`](../configs/parser.py)

Manages pipeline configuration from `config.txt`, including:
- MDRNN model parameters
- NST preprocessing settings  
- Output directory specifications
- Processing flags and options

### Processing Path Management
**File:** `src/cell/processing_path_manager.py`  
**Class:** [`ProcessingPathManager`](../cell/processing_path_manager.py)

Handles file system organization:
- Subject/session directory structure
- Input image locations
- Output directory management
- Montage mosaic loading/saving

### Error Handling and Logging

The pipeline includes error handling:

```python
try:
    postprocessing_pipeline.go()
except Exception as e:
    traceback.print_exc()
    logging.critical("CRITICAL ERROR OCCURRED!")
    logging.exception("Exception occurred", exc_info=True)
    logging.critical("STOPPING PIPELINE")
```

**Logging Features:**
- Environment-based log levels (DEBUG for development, WARNING for production)
- File-based logging to `logs/app.log`
- Console output for interactive debugging
- Exception tracking with full stack traces

## Execution Modes

### Single Subject Processing
Process individual subject by providing specific DataFrame:
```python
df_single = pd.DataFrame({
    'name': ['path/to/Subject001/Session001'],
    'mdrnn': [True],
    'montaging_corrected': [True]
})
main(df_single)
```

### Batch Processing  
Process entire cohort by calling without parameters:
```python
main()  # Processes all subjects in base directory
```

### Configuration-Driven Processing
Control pipeline execution via `config.txt` flags:
- `do_mdrnn_cell_detection = True`
- `do_montaging = True` 
- `do_registring = False`

## Dependencies and Prerequisites

### Required Pipeline Steps (must run before MDRNN):
1. **Montaging**: Creates spatial coordinate system
2. **Montage Correction**: Manual GUI corrections if needed

### Optional Pipeline Steps:
1. **Image Registration**: For multi-session analysis
2. **SSIM Processing**: For improved montage quality

### System Dependencies:
- PyTorch installation for neural network models
- CUDA support for GPU acceleration (recommended)
- Sufficient memory for processing high-resolution retinal images

## Output Structure

For each processed subject, the MDRNN pipeline generates results in the subject's directory:

```
Subject###/Session###/
├── postprocessed_atms_single/       # MDRNN cell detection results
├── montaged_corrected/             # Required input montage
├── raw_run/                        # Intermediate processing files  
└── checkpoint/                     # Model checkpoints and states
```

## Integration with Other Pipelines

The NST/MDRNN pipeline integrates with:

- **[Montaging Pipeline](montage_pipeline.md)**: Provides spatial coordinate system
- **[Registration Pipeline](registration_pipeline.md)**: Multi-session analysis capability
- **[Density Analysis Pipeline](density_analysis_pipeline.md)**: Uses MDRNN cell counts for density calculation
- **[ATMS Pipeline](atms_pipeline.md)**: Alternative cell detection method for comparison

## Next Steps

The MDRNN pipeline output feeds into downstream analysis components:
- **[Cell Density Calculator](cone_density_calculator.md)**: Calculation and spatial mapping
- **[Validation Tools](validation_tools.md)**: Comparison with manual cell counts
- **[Layer Thickness Analysis](layer_thickness_analysis.md)**: Integration with OCT data
- **[Statistical Analysis](density_statistics.md)**: Analysis across subject cohorts

---

**Note**: This documentation covers the top-level pipeline orchestration. For detailed implementation of the MDRNN algorithm, NST preprocessing, and cell detection methods, see the [`MDRNNPipelineManager`](../cell/cell_detection/mdrnn/mdrnn_pipeline_manager.md) documentation.

## Dependencies


### Pipeline Dependencies
- **[Processing Path Manager](processing_path_manager.md)**: File system organization and path management
- **[Configuration Parser](config_parser.md)**: Pipeline configuration and parameter management
- **[Montage Pipeline](montage_pipeline.md)**: Spatial coordinate system establishment (must run before MDRNN)
- **[Montage Mosaic](montage_mosaic.md)**: Reconstructed retinal image data structure


### Optional Dependencies
- **[Image Registration](image_registration.md)**: Multi-session alignment
- **[SSIM Processing](ssim_processing.md)**: Montage quality improvement
- MATLAB Runtime (if using MATLAB-based preprocessing components)