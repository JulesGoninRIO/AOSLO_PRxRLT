import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from torch import Value
# import importlib

# os.chdir(r'C:\Users\CordonnierA\Desktop\Repo\aoslo_pipeline')
current_file = os.path.abspath(__file__)
parent_of_parent =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.append(parent_of_parent)
from src.cell.processing_path_manager import ProcessingPathManager, Path
from src.cell.montage.montage_mosaic import MontageMosaic
from src.cell.analysis.cone_density_calculator import ConeDensityCalculator
from src.cell.layer.layer_thickness_calculator import LayerThicknessCalculator
from src.cell.analysis.density_layer_analysis import DensityLayerAnalysis
from src.cell.layer.layer_collection import LayerCollection
from src.cell.analysis.normal_comparison_handler import NormalComparisonHandler
from src.cell.montage.montage_mosaic_builder import CorrectedMontageMosaicBuilder


class DensityAnalysisPipelineManager():
    """
    Manage the cell detection pipeline.

    This class manages the cell detection pipeline, including initializing the necessary
    components and running the pipeline.

    :param processing_path_manager: The manager for processing paths.
    :type processing_path_manager: ProcessingPathManager
    :param montage_mosaic: The montage mosaic to be processed.
    :type montage_mosaic: MontageMosaic
    """
    def __init__(self, processing_path_manager: ProcessingPathManager, montage_mosaic: MontageMosaic, step: float = 0.1):
        self.path_manager = processing_path_manager
        self.montage_mosaic = montage_mosaic
        self.step = step
        self.debugpath = str('P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\OSDebug')




    def plot_thickness_curves_by_layer(self, layers, patient_id: str) -> None:
        """
        Plot the individual thickness curves (from the arrays) and the mean curve for each layer.
        Each plot is saved in a subfolder corresponding to the given patient (patient_id), under self.debugpath.
        NaN values are handled using a masked array so that lines break at missing data.

        :param layers: The dictionary of layer objects.
        :param patient_id: The patient identifier, used to create a subfolder under self.debugpath.
        """
        # Create the patient-specific output directory if it doesn't exist.
        output_dir = os.path.join(self.debugpath, patient_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate over each layer in the pipeline
        for layer_name, layer in layers.items():
            # Check if the layer has the required attributes; skip if not
            if not (hasattr(layer, 'thickness_per_distance_X') and hasattr(layer, 'mean_thickness_per_distance_X')):
                continue

            # Get the dictionaries with thickness data.
            thickness_dict = layer.thickness_per_distance_X  # e.g. {distance: [thickness1, thickness2, ...]}
            mean_dict = layer.mean_thickness_per_distance_X    # e.g. {distance: mean_thickness}
            
            # Get a sorted list of distances (keys)
            distances = sorted(thickness_dict.keys())
            
            # Determine the maximum number of measurements across all distances
            min_measurements = min(len(vals) for vals in thickness_dict.values())
            
            # Prepare a list of curves.
            # For each measurement index, build a list of (distance, thickness) points.
            curves = []
            for idx in range(min_measurements):
                curve_distances = []
                curve_values = []
                for d in distances:
                    values = thickness_dict[d]
                    # Only include if there is a measurement for this index at this distance
                    if idx < len(values):
                        curve_distances.append(d)
                        curve_values.append(values[idx])
                # Add the curve if it has any points
                if curve_distances:
                    curves.append((curve_distances, curve_values))
            
            # Start plotting
            plt.figure(figsize=(10, 6))
            
            # Plot individual curves with lower alpha for transparency.
            # Use a masked array so that NaNs create breaks in the line.
            for curve in curves:
                xs = np.array(curve[0])
                ys = np.array(curve[1])
                masked_ys = np.ma.masked_invalid(ys)
                plt.plot(xs, masked_ys, color='blue', alpha=0.3, linewidth=1)
            
            # Plot the mean curve with full opacity.
            mean_x = sorted(mean_dict.keys())
            mean_y = [mean_dict[d] for d in mean_x]
            # If there are any NaNs in the mean values, mask them as well.
            plt.plot(mean_x, np.ma.masked_invalid(mean_y), color='red', alpha=1.0, linewidth=2, label='Mean Thickness')
            
            # Add labels, title, legend, and grid.
            plt.xlabel("Distance")
            plt.ylabel("Thickness")
            plt.title(f"Thickness Curves for Layer: {layer_name} (Patient: {patient_id})")
            plt.legend()
            plt.grid(True)
            
            # Save the plot in the patient-specific output directory.
            save_path = os.path.join(output_dir, f"{layer_name}_thickness_curves.png")
        
            plt.savefig(save_path)
            plt.close()

    
    def run(self, do_layer: bool = False, from_csv: bool = False, from_csv_raw: bool = False, to_csv_dens: bool = True, subject: str = None):
        if from_csv:
            print("DEBUG: selected to get densities from csv")
            import pandas as pd
            import numpy as np
            from src.cell.analysis.density import Density
            df = pd.read_csv(self.path_manager.path / self.path_manager.density.path.name / 'densities.csv', delimiter=';', index_col=False)
            # print("DEBUG:", df.head())
            
            fno = lambda x: x[~np.isnan(x[:,1])] # filter nan out

            # print ("DEBUG: X: ", dict(fno(df[['ecc', 'dens_X']].values)))
            
            densities = Density(
                
                X=dict(fno(df[['ecc', 'dens_X']].values)),
                Y=dict(fno(df[['ecc', 'dens_Y']].values)),
                X_smoothed=dict(fno(df[['ecc', 'dens_smthd_X']].values)),
                Y_smoothed=dict(fno(df[['ecc', 'dens_smthd_Y']].values)),
                X_fitted=dict(fno(df[['ecc', 'dens_fit_X']].values)),
                Y_fitted=dict(fno(df[['ecc', 'dens_fit_Y']].values))
            )
        else:
            # print("DEBUG calculating densities with the following parameters")
            # print(f"DEBUG to csv: {to_csv_dens}")
            # print(f"DEBUG from csv raw:{from_csv_raw}")
            # print(f"DEBUG montage_mosaic: {self.montage_mosaic.__repr__}")
            # print(f"DEBUG path manager:{self.path_manager.path}")
            densities = ConeDensityCalculator(self.path_manager, self.montage_mosaic, self.step).get_densities_by_yellott(from_csv=from_csv_raw, to_csv=to_csv_dens)

        if do_layer:
            print("DEBUG: selected to calculate layers anew from extracted data")
            layer_collection = LayerCollection(self.path_manager)
            print("DEBUG: getting padded layers")
            layers_, scale, center = layer_collection.get_padded_layers()
            print("DEBUG: getting layer thicknesses")
            layers = LayerThicknessCalculator(self.path_manager, layers_, scale, center).get_layer_thicknesses()
            dla = DensityLayerAnalysis(self.path_manager, densities, layers)
            dla.process_layer_densities()
            
            #plot the thickness curves
            self.plot_thickness_curves_by_layer(dla.layers, patient_id=str(subject))

            dla.write_results()
            # NormalComparisonHandler().compare_to_normal(self.path_manager)

from time import time
import pickle
def load_pickle(path):
    start = time()
    with open(path, 'rb') as f:
        mosaic = pickle.load(f)
    print(f'Time to load pickle: {time() - start:.1f}s')
    return mosaic

def load_mosaic(path_manager):
    start = time()
    mosaic = CorrectedMontageMosaicBuilder(path_manager).build_mosaic()
    mosaic.save()
    print(f'Time to load & build mosaic: {time() - start:.1f}s')
    return mosaic

def main(path: str | None = None, do_montage: bool = True, **kwargs):
    print('Script started')
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger()
    # logger.setLevel(logging.ERROR)
    if path is None:
        path = r'C:\Users\CordonnierA\Documents\AIRDROP\data\run\Subject104\Session492'
    path = Path(sys.argv[1] if len(sys.argv) > 1 else path)
    path = Path(path)

    print("DEBUG: path passed as path_manager:", path)
    path_manager = ProcessingPathManager(path)
    print("DEBUG: path passed as path_manager after assignement:", path_manager.path)
    if do_montage:
        path_manager.montage.initialize_montage()
        if (pkl_path := path_manager.montage.corrected_path / 'mosaic.pkl').exists():
            try:
                mosaic = load_pickle(pkl_path)
                assert mosaic._mosaic_cache is not None
            except KeyboardInterrupt:
                raise
            except:
                mosaic = load_mosaic(path_manager)
        else:
            mosaic = load_mosaic(path_manager)
    else: 
        mosaic = load_mosaic(path_manager)
    DensityAnalysisPipelineManager(path_manager, mosaic, step=0.1).run(**kwargs)

if __name__ == '__main__':
    
    # p1= r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject107\Session503'
    # p2 = r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject93\Session434'
    # ids = [107,93]
    
    
    # for i, patient in enumerate([p1,p2]):
    #     print(f"Processing {patient}")
    #     main(patient, do_montage = False, do_layer = True, from_csv = True, from_csv_raw = True, to_csv_dens = True, subject = ids[i])

    # raise ValueError("Stop here")


    base_path = Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results')
    import numpy as np
    import pandas as pd
    # processed = np.loadtxt('processed', dtype=int).tolist()
    # NCH = NormalComparisonHandler()
    for sub_path in base_path.iterdir():
        if 'Subject' not in sub_path.name or not sub_path.is_dir():
            continue
        subject = int(sub_path.name.lstrip('Subject'))
        print(f'Processing subject {subject:>3} {80 * '='}')
        for ses_path in sub_path.iterdir():
            if 'Session' not in ses_path.name or not ses_path.is_dir():
                continue
            session = int(ses_path.name.lstrip('Session'))
            if not (ses_path / 'density_analysis').exists() or not (ses_path / 'montaged_corrected').exists():
                print(f'Ignoring session {session} for subject {subject}')
                continue
            try:
                main(ses_path, do_montage = False, do_layer = True, from_csv = True, from_csv_raw = False, to_csv_dens = True, subject = subject)
                # NCH.compare_to_normal(ProcessingPathManager(ses_path))
            except Exception as e:

                print(f'Error processing {sub_path.name} {ses_path.name}: {repr(e)}')
                continue
            finally:
                print(f'Finished processing {sub_path.name}')
