# Global Density Analysis step

This step allows to analyze the densities and layer thicknesses of all subjects.

## Parameters explanation

1. **Global Density Analysis Directory**: The name of the directory where the results of the Global Density Analysis will be found (will be *base_path/global_density_analysis_dir*).

## Parameters from other steps that matters when computing Global Density Analysis

You should have done the Density Analysis completely for the subjects you want to include in the Global Density Analysis. You will have a file in the output directory called *subjects_taken_into_account_for_normals.txt* where you will have the list of all the subjects taken into account to have all the graphs shown.

## Run the Global Density Analysis step

1. Run via the **InputGUI**: Go to the specific directory via: `cd aoslo_pipeline/src/InputGUI` And run: `python main.py`. Then make sure the tickbox **Global Analysis** is checked, that you have specified the parameters you want in the **GLOBAL ANALYSIS** subsection and that you have selected the correct input folder. Then just press the **Run pipeline!** and follow the instructions.

2. Run via the *config.txt*: Write in the config file that can be found in *aoslo_pipeline/src/PostProc_Pipe/Configs/config.txt* the specific parameters that you want to run, especially everything under the [Density] section (*__do_global_analysis* must be set to **True** to run this step of the pipeline). Then go to the specific folder: `cd aoslo_pipeline/src/PostProc_Pipe` and run `python Start_PostProc_Pipe.py`.

## Substeps of Global Density Analysis and what to find in the output folder

1. First, results for all subjects in the base directory will be searched for the resulting CSV file. If the file does not exists, the subject will not be taken into account. The resulting CSV file either contain only densities (if layer thickness has not been selected) or densities and layer thicknesses. The subjects taken into account to compute global densities are stored in *subjects_taken_into_account_for_normals.txt*.

2. Raw Median densities (*raw_densities_axis.png*) and layer thicknesses (*raw_layer_name_axis.png*) will be plot, as well as a smoothed version (*smoothed_densities_axis.png* and *smoothed_layer_name_axis.png*) (for smoothing, see below), the median and errorbars (*median_errorbar_densities_axis.png* and *median_errorbar_layer_name_axis.png*) and a single plot with all the layer thicknesses compared to the densities for each axis (*layer_compared_to_thickness_axis.png*) (axis is replaced by x or y depending on axis).

3. Then single layer thicknesses will be plotted against densities for each axis, with a raw (*layer_name.png*) and a smoothed version (*layer_name_smoothed.png*).

4. Gather the spearman correaltions results between densities and layer thicknesses for each axis for subjects were resulting layer have been found. The spearman correlations are compute as the correlation between the first part of the density (from ~-10 to 0) and the first part of the layer thickness (from ~-10 to 0) for most of the layer thicknesses (Retinal Nerve Fiber Layer (RNFL), Choroidal Vascularity Index (CVI), Choriocapillaris and Choroidal Stroma (CC+CS = Choroid), Outer Nuclear layer (ONL), Photoreceptors + Retinal Pigment Epithelium (PR+RPE)). But for some layers (Inner Nuclear Layer and Outer Plexiform Layer (INL+OPL), Ganglion Cell Layer and Inner Plexiform Layer (GCL+IPL)), the spearman correlation is only computed for the first part of the curve because there is a hole in the center of the eye.

5. Plot the betas and p-values of the correlations gathered before for each axis (*betas_layer_name_axis.png*, *p_values_layer_name_axis.png*) and a violin-plot gathering the betas for p-values smaller than 0.05 (*spearman_correlation_for_axis.png*), where the percentage written aside the layer name is the percentage of p-values smaller than 0.05 for the specific layer. Also raw plots of betas and p-values are outputed, as well as a CSV file to read the results.

6. Compare the median density values with the median Choroid (Choriocapillaris and Choroidal Stroma) thickness in the very center of the eye (-4° to 4°) by plotting datapoints in *choroid_axis_median.png* (with plots for mean and std also.). The results are also saved in a CSV file for further analysis.

7. Plot the differences between the x and y dimensions for layer thicknesses and densities (*densities_x_vs_y.png* and *layer_name_x_vs_y.png*) and save valeus in a CSV file (*x_vs_y.xlsx*).

8. Write the median and standard deviation results of the subjects densities and layer thicknesses in a CSV file (*results_median_std.xlsx*)

9. Compare the densities and layer thicknesses to baseline characteristics of the subjects (age, axial length, spherical equivalence, sexe and date of visits) and save the results in a CSV file and plot the smoothed plot results in *characteristic_layer_name_axis.png* (where characteristic is replaced ba age, axial length, ... and layer_name by densities if we compare densities). A CSV file is also creatd for each characteristic with median and standard deviation *characteristic_median_std.png*.

10. Optionnally, you can compare the too small areas that give too big densities count (functions too_big_areas_analysis and too_big_areas_analysis_scatter) that will output plots as *too_big_scatter_separated_and_thresholded.png* to analyze the densities compared to the number of pixels taken into account when computing densities.
