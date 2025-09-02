from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import json
# import logging
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt


from src.cell.analysis.constants import MM_PER_DEGREE
from src.cell.layer.layer import Layer
from src.cell.layer.helpers import (
    LAYER_NAMES_TO_INCLUDE,
    LAYER_NAMES,
    gaussian_filter_nan,
    get_cube_center,
    modify_name
)
from src.cell.processing_path_manager import ProcessingPathManager
from src.plotter.layer_plotter import LayerPlotter


class LayerCollection:
    """
    A class used to manage and process layers of thickness data.

    This class handles loading, preprocessing, and visualization of layer thickness data
    from retinal scans.
    """

    def __init__(self, path_manager: ProcessingPathManager):
        """
        Initialize the LayerCollection with a path manager.

        :param path_manager: ProcessingPathManager instance containing path information
        :type path_manager: ProcessingPathManager
        """
        self.layer_thickness_path = path_manager.density.layer_path
        self.right_eye = path_manager.right_eye
        self.subject_id = path_manager.subject_id
        self.layers = {name: Layer(name) for name in LAYER_NAMES}
        self.spacing: Optional[Dict[str, float]] = None
        self.plotter: Optional[LayerPlotter] = None

    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess the layer thickness and CVI data.

        Loads data from JSON files, preprocesses it, and initializes the plotter.
        Sets the scale attribute and populates layer data.

        :raises ValueError: If spacing information is missing in the data
        """
        # raise ValueError("DEBUG: loading and preprocessing data, called by LayerCollection in file layer_collection.py")
        print("DEBUG: loading and preprocessing data, called by LayerCollection in file layer_collection.py")

        try:
            thicknesses = self._load_json('layer_thickness.json')
            cvis = self._load_json('layer_cvis.json')
            ospr = self._load_json('layer_os.json')
            self.spacing = thicknesses.get('spacing', None)
            if self.spacing is not None:
                print("DEBUG: spacing is not None, preprocessing data")
                self.plotter = LayerPlotter(self.layer_thickness_path, self.spacing)
                self._preprocess_data(thicknesses, cvis, ospr)
            else:
                raise ValueError("Missing 'spacing' in layer thickness data.")
        except Exception as e:
            print(f'Error loading or processing data: {e}')
            # logging.error(f'Error loading or processing data: {e}')

    def _load_json(self, filename: str) -> Dict:
        """
        Load a JSON file and return its contents as a dictionary.

        :param filename: Name of the JSON file to load
        :type filename: str
        :return: Dict containing the JSON file contents
        :rtype: Dict
        """
        return json.load((self.layer_thickness_path / filename).open())

    def _preprocess_data(self, thicknesses: Dict, cvis: Dict, ospr: Dict) -> None:
        """
        Preprocess the layer thickness and CVI data.

        :param thicknesses: Dictionary containing layer thickness data
        :type thicknesses: Dict
        :param cvis: Dictionary containing CVI data
        :type cvis: Dict
        :param ospr: Dictionary containing OS layer thickness data
        :type ospr: Dict
        """
        # make sure that scans are sorted by their index
        c_scan = OrderedDict(sorted(next(iter(thicknesses.values())).items(), key=lambda x: x[0]))
        cvi_c_scan = OrderedDict(sorted(next(iter(cvis.values())).items(), key=lambda x: x[0]))
        os_c_scan = OrderedDict(sorted(next(iter(ospr.values())).items(), key=lambda x: x[0]))

        # populate layer thickness data
        lengths = []
        for b_scan_id, b_scan in c_scan.items():
            for layer_name, layer_v in b_scan.items():
                name = modify_name(layer_name)
                if name in LAYER_NAMES_TO_INCLUDE:
                    self.layers[name].append_thickness(layer_v['vector'])
                    lengths.append(len(layer_v['vector']))
            # populate OS layer thickness data
            length_PRRPE = len(b_scan['PR_RPE']['vector'])
            if b_scan_id in os_c_scan:
                OS_layer = os_c_scan[b_scan_id]['vector']
                OS_layer = np.array(OS_layer, dtype=float)
                OS_layer[np.isnan(OS_layer)] = 0
                length_OS = OS_layer.shape[0]


                #

                # print("DEBUG: changing length of OS layer, max value before interpolation: ", np.max(OS_layer))

                # print("DEBUG: changing length of OS layer, min value before interpolation: ", np.min(OS_layer))
                # print("DEBUG: changing length of OS layer, mean value before interpolation: ", np.mean(OS_layer))
                #initailizing plot: we are plotting the OS values from before and after interpolation in the same plot to see the difference
                # print("DEBUG: PLOTTING VALUE DISTRIBUTION OF OS LAYER")
                # fig, ax = plt.subplots(ncols=2, figsize=(5, 5))  # Create two subplots side by side

                # # Original OS Layer Plot
                # ax[0].plot(OS_layer, label="Original OS Layer", )
                # ax[0].set_title("OS layer values before interpolation")
                # ax[0].legend()





                # because CohortBuilder tilts the B-scan before computing the thicknesses,
                # the segmented layers are typically longer or shorter than the original B-scans.
                # To ensure that the OS layer (that not from CohortBuilder) has the same length,
                # we stretch it (linearly, as a rotation is a linear transformation anyway)
                # to match the PR+RPE layer length; should be good enough for our purposes
                # assert length_OS - 1 <= length_PRRPE, f'OS layer length ({length_OS}) is greater than PR+RPE layer length ({length_PRRPE}) for B-scan {b_scan_id}'
                
                original_indices = np.arange(length_OS)
                stretched_indices = np.linspace(0, length_OS - 1, length_PRRPE)
                stretched_OS_layer = np.interp(stretched_indices, original_indices, OS_layer)
                stretched_OS_layer[stretched_OS_layer == 0] = np.nan
                assert len(stretched_OS_layer) == length_PRRPE
                self.layers['OS'].append_thickness(stretched_OS_layer.tolist())
                # print(f"DEBUG: changing length of OS layer {b_scan_id}, max value after interpolation: ", np.nanmax(stretched_OS_layer))

                # print("DEBUG: changing length of OS layer, min value after interpolation: ", np.nanmin(stretched_OS_layer))
                # print("DEBUG: changing length of OS layer, mean value after interpolation: ", np.nanmean(stretched_OS_layer))

                # print("DEBUG: PLOTTING VALUE DISTRIBUTION OF OS LAYER AFTER INTERPOLATION")

                # # Find closest matching indices for comparison
                # matching_indices = np.round(stretched_indices).astype(int)
                # matching_indices = np.clip(matching_indices, 0, length_OS - 1)  # Ensure indices are in range
                # OS_layer_matched = OS_layer[matching_indices]  # Extract corresponding values

                # Stretched OS Layer and matched OS Layer Plot
                # ax[1].plot(stretched_OS_layer, label="Stretched OS Layer")
                # ax[1].plot(matching_indices, OS_layer_matched, 'rx', label="Matching OS Layer Points")  # Highlight matching points
                # ax[1].set_title("OS layer values after interpolation")
                # ax[1].legend()

                # plt.tight_layout()  
                # plt.show()


            else:
                # print("DEBUG: adding nan values to OS layer, calling append_thickness of class Layer")
                self.layers['OS'].append_thickness([np.nan] * length_PRRPE)

        # populate CVI data, handling None values correctly
        goal_length = int(np.mean(lengths))
        for b_scan in cvi_c_scan.values():
            # for CVI, the way the browse function of CohortExtractor works makes
            # it so that the beginning and end of the vector contain extra None 
            # values that need to be removed
            cvi_vector = self._remove_nones(b_scan['vector'], goal_length)
            self.layers['CVI'].append_thickness(cvi_vector)
            lengths.append(len(cvi_vector))

        # this will be used to pad the layers later
        self.max_length = int(np.max(lengths))

    def _remove_nones(self, layer_v: List[Optional[float]], goal_length: int) -> List[float]:
        """
        Remove None values from a layer vector and adjust its length.

        :param layer_v: List of thickness values that may contain None
        :type layer_v: List[Optional[float]]
        :param goal_length: Target length for the output list
        :type goal_length: int
        :return: List of float values with None values removed and length adjusted
        :rtype: List[float]
        """
        r_list = lambda s,e: [v if v is not None else 0 for v in layer_v[s:-e]]
        start = next(i for i, x in enumerate(layer_v) if x is not None)
        end = next(i for i, x in enumerate(reversed(layer_v)) if x is not None)
        if len(layer_v) - end - start < 1.08 * goal_length:
            # if not too far from the goal length
            return r_list(start, end)
        else:
            # print(len(layer_v), end, start, goal_length)
            s_left = len(layer_v) - end - goal_length
            nstart = min(start, s_left) if s_left > 0 else start
            s_right = len(layer_v) - start - goal_length
            nend = min(end, s_right) if s_right > 0 else end
            if 0.92 < (len(layer_v) - nend - nstart) / goal_length < 1.08:
                return r_list(nstart, nend)
            nstart = (len(layer_v) - goal_length) // 2
            nend = len(layer_v) - nstart - goal_length
            return r_list(nstart, nend)

    def get_padded_layers(self) -> Tuple[Dict[str, Layer], Dict[str, float], Tuple[int, int]]:
        """
        Get the padded layers ensuring uniform shape across all layers.

        :return: Tuple containing layers, spacing, and center peak coordinates
        :rtype: Tuple[Dict[str, Layer], Dict[str, float], Tuple[int, int]]
        """
        print("DEBUG: getting padded layers, called by LayerCollection in file layer_collection.py", flush=True)
        self.load_and_preprocess_data()

        # as all bscan may not have the same length (because the browse function
        # of CohortExtractor rotates the images a little, so the width varies),
        # we need to pad the layers to ensure they all have the same shape. The 
        # transformation done by CE is linear and symmetric about the center, so
        # we can simply pad the layers with the same number of pixels on each side
        get_padding = lambda n: ((self.max_length - n) // 2, self.max_length - n - (self.max_length - n) // 2)

        for layer in self.layers.values():
            if not layer.is_empty():
                 layer.padded_thickness_values = self._invert_layer([
                    np.pad(
                        np.array(layer_values),
                        get_padding(len(layer_values)),
                        'constant'
                    )
                    for layer_values in layer.thickness_values
                ])

        padded_layers = {name: layer.padded_thickness_values for name, layer in self.layers.items() if not layer.is_empty()}

        assert all([layer.shape == next(iter(padded_layers.values())).shape for layer in padded_layers.values()])

        center = get_cube_center(self.subject_id, next(iter(padded_layers.values())).shape)

        #flips the center as well

        arraycenterx= (len((self.layers["OS"].padded_thickness_values))//2)

    
        centerx = arraycenterx - (center[0]-arraycenterx)
        if not self.right_eye:
            arraycentery = (self.max_length)//2
            centery = arraycentery - (center[0]-arraycentery)
        else:
            centery = center[1]

        center = (centerx, centery)

        layer.centerx = centerx

        
        # center2 = get_center_peak(padded_layers['PhotoR+RPE'])
        self._plot_layers(padded_layers, center)

        return self.layers, self.spacing, center

    def _invert_layer(self, layer_values: List[np.ndarray]) -> np.ndarray:
        """
        Invert layer values to ensure consistent axis orientation.

        Ensures Y-axis is directed Superior->Inferior and X-axis is Temporal->Nasal. Initially, `layer_values` is a (nb_bscan, nb_ascan) matrix, 
        where the first axis is the Y-axis (indexed from Inferior to Superior)
        and the second axis is the X-axis (indexed from Temporal to Nasal for right eyes, and from Nasal to Temporal for left eyes). 

        :param layer_values: List of layer value arrays to invert
        :type layer_values: List[np.ndarray]
        :return: Inverted numpy array with correct orientation
        :rtype: np.ndarray
        """
        layer_values = np.flip(np.array(layer_values), axis=0)
        if not self.right_eye:
            layer_values = np.flip(layer_values, axis=1)
        return layer_values

    def _plot_layers(self, padded_layers: Dict[str, np.ndarray], center: Tuple[int, int]) -> None:
        """
        Plot the layers using Gaussian and maximum filters.

        :param padded_layers: Dictionary of padded layer arrays
        :type padded_layers: Dict[str, np.ndarray]
        :param center: Tuple containing center coordinates
        :type center: Tuple[int, int]
        """
        for layer_name, layer_list in padded_layers.items():
            if len(layer_list) > 0:
                layer = list(maximum_filter(np.array(layer_list), size=5))
                layer = np.array(layer)
                layer_5 = gaussian_filter_nan(layer, sigma=5)

                X = np.arange(0, len(layer_5[0]), 1)
                x_limits = [         - center[1]  * self.spacing['x'] / MM_PER_DEGREE, 
                    (len(layer_5[0]) - center[1]) * self.spacing['x'] / MM_PER_DEGREE ]
                # x_degrees = np.arange(math.floor(x_limits[0]), math.ceil(x_limits[1]), step=2)
                x_degrees = np.arange(-10, 11, step=2)
                x_number = center[1] + x_degrees * MM_PER_DEGREE / self.spacing['x']

                Y = np.arange(0, len(layer_5), 1)
                y_limits = [      - center[0]  * self.spacing['y'] / MM_PER_DEGREE,
                    (len(layer_5) - center[0]) * self.spacing['y'] / MM_PER_DEGREE ]
                # y_degrees = np.arange(math.floor(y_limits[0]), math.ceil(y_limits[1]), step=2)
                y_degrees = np.arange(-10, 11, step=2)
                y_number = center[0] + y_degrees * MM_PER_DEGREE / self.spacing['y']
                
                self.plotter.plot(layer_name, layer_5, [X, Y], [x_number, y_number], [x_degrees, y_degrees])

if __name__ == '__main__':
    # layer_thickness_path = Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject18\Session296\layer')
    layer_thickness_path = Path(r'C:\Users\CordonnierA\Documents\AIRDROP\data\run\Subject104\Session492\layer_new')
    layer_collection = LayerCollection(layer_thickness_path)
    # layer_collection.load_and_preprocess_data()
    layers, scale, peak = layer_collection.get_padded_layers()
    # import pickle
    # with open(r'C:\Users\BardetJ\Downloads\layers.pkl', 'wb') as f:
    #     pickle.dump(layers, f)
    print('oe')
