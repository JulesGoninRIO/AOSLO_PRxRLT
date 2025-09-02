window.pipelineData = {
  "nodes": [
    {
      "id": "pipeline_runner",
      "label": "PipelineRunner",
      "category": "entry_point",
      "description": "Main pipeline orchestrator. Coordinates all AOSLO postprocessing pipelines including MDRNN cell detection.",
      "details": "Central orchestrator reading configuration, managing processing flags, and coordinating execution of different analysis modules.",
      "doc_link": "docs/nst_mdrnn_pipeline.md",
      "critical": true
    },
    {
      "id": "mdrnn_pipeline_manager",
      "label": "MDRNNPipelineManager",
      "category": "entry_point",
      "description": "Central coordinator for MDRNN processing. Manages preprocessing, algorithm execution, and postprocessing.",
      "details": "Determines preprocessing strategy, coordinates execution sequence, and ensures proper data flow between components.",
      "doc_link": "docs/mdrnn_detailed.md#mdrnnpipelinemanager",
      "critical": true
    },
    {
      "id": "path_manager",
      "label": "ProcessingPathManager",
      "category": "utility",
      "description": "Manages file paths and directory structure for the pipeline.",
      "details": "Handles subject/session directory structure, input/output locations, and montage mosaic loading/saving.",
      "doc_link": "docs/processing_path_manager.md",
      "critical": true
    },
    {
      "id": "montage_mosaic",
      "label": "MontageMosaic",
      "category": "data_structure",
      "description": "Data structure holding reconstructed retinal mosaic from montaging pipeline.",
      "details": "Contains spatial coordinate system centered at fovea, montage elements, and transformation data.",
      "doc_link": "docs/montage_mosaic.md",
      "critical": true
    },
    {
      "id": "mdrnn_preprocessor",
      "label": "MDRNNPreProcessor",
      "category": "processor",
      "description": "Transforms full retinal images into analysis-ready patches through multi-stage enhancement.",
      "details": "Handles image segmentation, statistical corrections, histogram matching, and optionally Neural Style Transfer.",
      "doc_link": "docs/mdrnn_detailed.md#mdrnnpreprocessor",
      "critical": true
    },
    {
      "id": "mdrnn_algorithm",
      "label": "MDRNNAlgorithm",
      "category": "processor",
      "description": "Interfaces with external TensorFlow-based MDRNN cone detection algorithm.",
      "details": "Manages conda environment execution, file system organization, and coordinates external process execution.",
      "doc_link": "docs/mdrnn_detailed.md#mdrnnnalgorithm",
      "critical": true
    },
    {
      "id": "mdrnn_postprocessor",
      "label": "MDRNNPostProcessor",
      "category": "processor",
      "description": "Analyzes detection results to generate quality metrics and statistical summaries.",
      "details": "Applies Otsu thresholding, categorizes patches, generates visualization plots, and produces statistical analyses.",
      "doc_link": "docs/mdrnn_detailed.md#mdrnnpostprocessor",
      "critical": true
    },
    {
      "id": "preprocessing_params",
      "label": "MDRNNPreProcessingParams",
      "category": "data_structure",
      "description": "Configuration parameters for preprocessing methods and enhancement techniques.",
      "details": "Specifies method type, range processing, replacement strategies, and enhancement options (NST, histogram matching).",
      "doc_link": "docs/mdrnn_preprocessing_params.md",
      "critical": true
    },
    {
      "id": "patch_cropper",
      "label": "PatchCropper",
      "category": "utility",
      "description": "Segments full images into analysis patches for MDRNN processing.",
      "details": "Handles image segmentation into standardized patches with proper numbering and organization.",
      "doc_link": "docs/patch_cropper.md",
      "critical": false
    },
    {
      "id": "nst_pipeline_manager",
      "label": "NSTPipelineManager",
      "category": "processor",
      "description": "Coordinates complete Neural Style Transfer workflow including optimization execution and loss tracking.",
      "details": "Manages NST optimization iterations, records loss values, generates learning curve visualizations, and exports optimization data.",
      "doc_link": "docs/nst_pipeline.md#nstpipelinemanager",
      "critical": true
    },
    {
      "id": "nst_algorithm",
      "label": "NST",
      "category": "processor",
      "description": "Implements core Neural Style Transfer algorithm using VGG feature extraction and Gram matrix computation.",
      "details": "Performs iterative optimization with content and style loss computation, Adam optimizer updates, and gradient-based image enhancement.",
      "doc_link": "docs/nst_pipeline.md#nst",
      "critical": true
    },
    {
      "id": "vgg_model",
      "label": "VGG Model",
      "category": "data_structure",
      "description": "Pre-trained VGG network for multi-layer feature extraction in style transfer.",
      "details": "Provides convolutional feature maps at multiple scales for content preservation and style representation through Gram matrices.",
      "doc_link": "docs/vgg_model.md",
      "critical": true
    },
    {
      "id": "image_files",
      "label": "ImageFile Patches",
      "category": "data_structure",
      "description": "Individual image patches processed by the MDRNN algorithm.",
      "details": "Contains cropped image data, spatial coordinates, and preprocessing metadata for analysis.",
      "doc_link": "docs/image_file.md",
      "critical": true
    },
    {
      "id": "algorithm_markers",
      "label": "algorithmMarkers/",
      "category": "output",
      "description": "Directory containing CSV files with detected cone coordinates for each patch.",
      "details": "Individual coordinate files for each processed image patch with (x,y) cone positions.",
      "doc_link": "docs/outputs.md#algorithm-markers",
      "critical": true
    },
    {
      "id": "center_csv",
      "label": "center.csv",
      "category": "output",
      "description": "Raw algorithm output file with detection results before reorganization.",
      "details": "Contains image paths and center coordinates from MDRNN algorithm execution.",
      "doc_link": "docs/outputs.md#center-csv",
      "critical": true
    },
    {
      "id": "raw_run_dir",
      "label": "raw_run/",
      "category": "output",
      "description": "Directory containing preprocessed image patches for debugging and quality control.",
      "details": "Intermediate TIFF files showing preprocessing results for troubleshooting and analysis.",
      "doc_link": "docs/outputs.md#raw-run",
      "critical": false
    },
    {
      "id": "analysis_plots",
      "label": "Analysis Plots",
      "category": "output",
      "description": "Quality assessment visualizations including Otsu threshold analysis and NST learning curves.",
      "details": "PNG files showing detection statistics, quality distributions, preprocessing effectiveness, and NST optimization progress.",
      "doc_link": "docs/outputs.md#analysis-plots",
      "critical": false
    },
    {
      "id": "quality_metrics",
      "label": "Quality Metrics",
      "category": "output",
      "description": "Statistical summaries categorizing patches as 'good' or 'bad' based on detection success.",
      "details": "Threshold-based classification results and detection performance statistics.",
      "doc_link": "docs/outputs.md#quality-metrics",
      "critical": true
    }
  ],
  "edges": [
    {
      "source": "pipeline_runner",
      "target": "path_manager",
      "label": "initializes",
      "type": "creates"
    },
    {
      "source": "pipeline_runner",
      "target": "mdrnn_pipeline_manager",
      "label": "creates",
      "type": "creates"
    },
    {
      "source": "pipeline_runner",
      "target": "montage_mosaic",
      "label": "loads",
      "type": "dependency"
    },
    {
      "source": "mdrnn_pipeline_manager",
      "target": "mdrnn_preprocessor",
      "label": "creates",
      "type": "creates"
    },
    {
      "source": "mdrnn_pipeline_manager",
      "target": "mdrnn_algorithm",
      "label": "creates",
      "type": "creates"
    },
    {
      "source": "mdrnn_pipeline_manager",
      "target": "mdrnn_postprocessor",
      "label": "creates",
      "type": "creates"
    },
    {
      "source": "mdrnn_preprocessor",
      "target": "preprocessing_params",
      "label": "uses",
      "type": "dependency"
    },
    {
      "source": "mdrnn_preprocessor",
      "target": "patch_cropper",
      "label": "uses",
      "type": "delegates"
    },
    {
      "source": "mdrnn_preprocessor",
      "target": "nst_pipeline_manager",
      "label": "optionally_uses",
      "type": "delegates",
      "conditional": true
    },
    {
      "source": "nst_pipeline_manager", 
      "target": "nst_algorithm",
      "label": "creates",
      "type": "creates"
    },
    {
      "source": "nst_algorithm",
      "target": "vgg_model", 
      "label": "uses",
      "type": "dependency"
    },
    {
      "source": "nst_pipeline_manager",
      "target": "analysis_plots",
      "label": "generates_learning_curves",
      "type": "produces"
    },
    {
      "source": "mdrnn_preprocessor",
      "target": "image_files",
      "label": "produces",
      "type": "produces"
    },
    {
      "source": "patch_cropper",
      "target": "image_files",
      "label": "creates",
      "type": "produces"
    },
    {
      "source": "mdrnn_algorithm",
      "target": "center_csv",
      "label": "generates",
      "type": "produces"
    },
    {
      "source": "mdrnn_algorithm",
      "target": "algorithm_markers",
      "label": "creates",
      "type": "produces"
    },
    {
      "source": "mdrnn_algorithm",
      "target": "raw_run_dir",
      "label": "populates",
      "type": "produces"
    },
    {
      "source": "mdrnn_postprocessor",
      "target": "quality_metrics",
      "label": "generates",
      "type": "produces"
    },
    {
      "source": "mdrnn_postprocessor",
      "target": "analysis_plots",
      "label": "creates",
      "type": "produces"
    }
  ],
  "categories": {
    "entry_point": {
      "color": "#e74c3c",
      "label": "Entry Points",
      "description": "Main pipeline entry points and orchestrators"
    },
    "processor": {
      "color": "#3498db", 
      "label": "Core Processors",
      "description": "Main processing components that transform data"
    },
    "data_structure": {
      "color": "#f39c12",
      "label": "Data Structures", 
      "description": "Classes that hold and organize pipeline data"
    },
    "utility": {
      "color": "#27ae60",
      "label": "Utilities",
      "description": "Support classes for path management, visualization, and configuration"
    },
    "analysis": {
      "color": "#9b59b6",
      "label": "Analysis Tools",
      "description": "Components that perform statistical analysis and quality assessment"
    },
    "output": {
      "color": "#e67e22",
      "label": "Output Files",
      "description": "Generated results, plots, and data files from the pipeline"
    }
  }
};