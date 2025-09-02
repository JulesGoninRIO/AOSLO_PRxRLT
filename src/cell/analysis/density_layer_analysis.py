import pickle
import logging
from typing import Tuple, Dict, List
from pathlib import Path
import json
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from src.cell.analysis.density import Density
from src.cell.analysis.helpers import replace_nans
from src.cell.analysis.result_writer import ResultWriter
from src.cell.layer.layer import Layer
from src.cell.layer.helpers import gaussian_filter_nan
from src.cell.processing_path_manager import ProcessingPathManager
from src.shared.helpers.direction import Direction
from src.configs.parser import Parser
from src.plotter.density_layer_plotter import DensityLayerPlotter

class DensityLayerAnalysis:
    """
    A class used to analyze the density layers.

    :param path_manager: The manager for processing paths.
    :type path_manager: ProcessingPathManager
    :param densities: The densities data.
    :type densities: dict
    :param layers: The layers data.
    :type layers: dict
    """

    def __init__(self, path_manager: ProcessingPathManager, densities: Density, layers: Dict[str, Layer]):
        """
        Initialize the DensityLayerAnalysis with the given path manager, densities, and layers.

        :param path_manager: The manager for processing paths.
        :type path_manager: ProcessingPathManager
        :param densities: The densities data.
        :type densities: Dict[str, int]
        :param layers: The layers data.
        :type layers: Dict[str, Layer]
        """
        self.path_manager = path_manager
        self.densities = densities
        self.layers_to_avoid = ['IRF', 'PED', 'SRF', 'UNKNOWN']
        # filter out layers to avoid from the layers dictionary
        self.layers = {layer_name: layer for layer_name, layer in layers.items() if layer_name not in self.layers_to_avoid}
        self.step = 0.1
        self.round_step = round(self.step * 10)
        growing_factor = Parser.get_density_growing_factor()
        self.growing_factor = lambda x: growing_factor[0] * x + growing_factor[1]
        self.layers_to_separate_spearman = ['GCL+IPL', 'INL+OPL']
        self.spearmans = {}
        self.plotter = DensityLayerPlotter(self.path_manager.density.path)

    def process_layer_densities(self) -> None:
        """
        Process the layer densities.

        This method processes the densities for each layer in both x and y directions.
        It smooths the densities and layers, plots the thicknesses to densities, and analyzes each layer.

        :return: None
        """
        for direction in Direction:
            self.smooth_layers(direction)
            self.plotter.plot_thicknesses_to_densities(self.layers, self.densities, direction, self.path_manager.subject_id)
            for layer_name in self.layers.keys():
                self.analyze_layer(layer_name, direction)
            # save the spearman results to txt
            (self.path_manager.density.path / f'spearmans_{direction.value}.txt').open('w').write(json.dumps(self.spearmans, indent=4))
            self.spearmans = {}

    def write_results(self) -> None:
        """
        Write the density results to a file.

        This method writes the density results to a file, avoiding specified layers.

        :return: None
        """
        ResultWriter(self.path_manager, self.layers_to_avoid, self.step, self.round_step).write_density_results(self.densities, self.layers)


    def analyze_layer(self, layer_name: str, direction: Direction) -> None:
        """
        Analyze the specified layer.

        This method analyzes the specified layer by calculating Spearman's rank correlation coefficients
        for the layer's thickness and density data. It also plots the results.

        :param layer_name: The name of the layer to analyze.
        :type layer_name: str
        :return: None
        """
        self.spearmans[layer_name] = {}
        mean_thickness_per_distance = getattr(self.layers[layer_name], f'mean_thickness_per_distance_{direction.value}')
        density = getattr(self.densities, f'{direction.value}_fitted') # raw values may be trash, use _fitted or _smoothed
        # make sure that everything is sorted by eccentricity
        eccs_t, thicknsss = map(np.array, zip(*sorted(mean_thickness_per_distance.items())))
        eccs_d, densities = map(np.array, zip(*sorted(density.items())))
        common_eccs = np.intersect1d(eccs_t, eccs_d)
        left_eccs = common_eccs[common_eccs < 0]
        right_eccs = common_eccs[common_eccs > 0]
        if layer_name in self.layers_to_separate_spearman:
            if len(eccs_t) == 0 or len(thicknsss) == 0: 
                return
            eccs_t, thicknsss = replace_nans(eccs_t, thicknsss)
            thicknsss_smooth, pos_peaks = self.smooth_and_find_peaks(thicknsss)
            if pos_peaks is None:
                # logging.info('Could not find the peaks')
                print('Could not find the peaks', layer_name, direction.value)
                self.calculate_spearmans_for_simple_regions(layer_name, mean_thickness_per_distance, density, left_eccs, right_eccs)
            else:
                limits = DensityLayerAnalysis.get_limits(eccs_t, pos_peaks)
                self.calculate_spearmans_for_separated_regions(layer_name, mean_thickness_per_distance, density, left_eccs, right_eccs, limits)
        else:
            self.calculate_spearmans_for_simple_regions(layer_name, mean_thickness_per_distance, density, left_eccs, right_eccs)
        self.plotter.plot_2d_thickness_vs_density(self.layers, self.densities, layer_name, direction)

        

    def smooth_and_find_peaks(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smooth the data and find peaks.

        This method smooths the given data using a Gaussian filter and finds the positive and negative peaks.
        It continues to smooth the data until exactly 2 positive peaks and 1 negative peak are found or
        until a maximum of 100 iterations is reached.

        :param y: The data to smooth and find peaks in.
        :type y: np.ndarray
        :return: A tuple containing the smoothed data and a list of arrays with the positions of positive and negative peaks.
        :rtype: Tuple[np.ndarray, List[np.ndarray]]
        """
        found = False
        i = 0
        while not found:
            y_smooth = gaussian_filter_nan(y, sigma=i)
            pos_peaks, _ = find_peaks(y_smooth)
            neg_peaks, _ = find_peaks(-y_smooth)
            # we need to have 2 bumps & 1 hole -> otherwise smoothen array more
            if len(pos_peaks) == 2 and len(neg_peaks) == 1:
                found = True
                break
            else:
                i += 1
            if i > 100:
                logging.info('Could not find the peaks')
                return y_smooth, None
        # print(f'finished with {i}')
        return y_smooth, pos_peaks

    @staticmethod
    def get_limits(eccs, pos_peaks):
        assert len(pos_peaks) == 2
        limits = [eccs[round(pos_peaks[0])], eccs[round(pos_peaks[1])]]
        limits = sorted(limits, key = lambda n:float(n))
        return limits

    def calculate_spearmans_for_separated_regions(self, layer_name: str, mean_thickn: Dict[float,float], densities: Dict[float,float], left_indices: List[int], right_indices: List[int], limits: Tuple[int]) -> None:
        
        a_first_part = [densities[indice] for indice in left_indices if indice <= limits[0]]
        a_second_part = [densities[indice] for indice in left_indices if indice >= limits[0]]
        a_third_part = [densities[indice] for indice in right_indices if indice <= limits[1]]
        a_fourth_part = [densities[indice] for indice in right_indices if indice >= limits[1]]

        b_first_part = [mean_thickn[indice] for indice in left_indices if indice <= limits[0]]
        b_second_part = [mean_thickn[indice] for indice in left_indices if indice >= limits[0]]
        b_third_part = [mean_thickn[indice] for indice in right_indices if indice <= limits[1]]
        b_fourth_part = [mean_thickn[indice] for indice in right_indices if indice >= limits[1]]

        # replace nans values in the arrays so that spearman is not nan
        a_first_part, b_first_part = replace_nans(a_first_part, b_first_part)
        a_second_part, b_second_part = replace_nans(a_second_part, b_second_part)
        a_third_part, b_third_part = replace_nans(a_third_part, b_third_part)
        a_fourth_part, b_fourth_part = replace_nans(a_fourth_part, b_fourth_part)
        
        try:
            self.spearmans[layer_name]['first_part'] = spearmanr(
                a_first_part, b_first_part)
            self.spearmans[layer_name]['second_part'] = spearmanr(
                a_second_part, b_second_part)
            self.spearmans[layer_name]['third_part'] = spearmanr(
                a_third_part, b_third_part)
            self.spearmans[layer_name]['fourth_part'] = spearmanr(
                a_fourth_part, b_fourth_part)
        except ValueError:
            # indices are missing
            self.spearmans[layer_name]['first_part'] = None
            self.spearmans[layer_name]['second_part'] = None
            self.spearmans[layer_name]['third_part'] = None
            self.spearmans[layer_name]['fourth_part'] = None

    def calculate_spearmans_for_simple_regions(self, layer_name: str, mean_thickn: Dict[float,float], densities: Dict[float,float], left_indices: List[int], right_indices: List[int]) -> None:
        """
        Calculate Spearman's rank correlation coefficients for simple regions.

        This method calculates Spearman's rank correlation coefficients for the left and right regions
        of the specified layer's thickness and density data.

        :param layer_name: The name of the layer to analyze.
        :type layer_name: str
        :param left_indices: The list of indices for the left region.
        :type left_indices: List[int]
        :param right_indices: The list of indices for the right region.
        :type right_indices: List[int]
        :return: None
        """
        a_left = [densities[indice] for indice in left_indices]
        a_right = [densities[indice] for indice in right_indices]
        b_left = [mean_thickn[indice] for indice in left_indices]
        b_right = [mean_thickn[indice] for indice in right_indices]

        # replace nans values in the arrays so that spearman is not nan
        a_left, b_left = replace_nans(a_left, b_left)
        a_right, b_right = replace_nans(a_right, b_right)
        try:
            self.spearmans[layer_name]['left'] = spearmanr(a_left, b_left)
            self.spearmans[layer_name]['right'] = spearmanr(a_right, b_right)
        except ValueError:
            # Not enough points in the arrays
            self.spearmans[layer_name]['left'] = None
            self.spearmans[layer_name]['right'] = None

    def smooth_layers(self, direction: Direction) -> None:
        """
        Smooth the layer thickness data using LOWESS.

        This method smooths the thickness data for each layer in the current direction using the LOWESS (Locally Weighted Scatterplot Smoothing) method.

        :return: None
        """
        for layer_name in self.layers.keys():
            if layer_name in self.layers_to_avoid:
                continue
            # Determine the attribute names based on direction
            mean_thickness_attr_name = f'mean_thickness_per_distance_{direction.value}'
            smoothed_thickness_attr_name = f'smoothed_thickness_per_distance_{direction.value}'

            # Access the correct thickness_per_distance attribute based on direction
            mean_thickness_per_distance = getattr(self.layers[layer_name], mean_thickness_attr_name)
            if len(mean_thickness_per_distance) == 0:
                continue

            # Perform LOWESS smoothing
            smoothed_values = sm.nonparametric.lowess(
                list(mean_thickness_per_distance.values()),
                list(mean_thickness_per_distance.keys()),
                frac=0.1
            )

            # Convert smoothed_values (which is an array of tuples) back to a dictionary
            smoothed_dict = dict(smoothed_values)

            # Update the layer's smoothed_thickness_per_distance attribute
            setattr(self.layers[layer_name], smoothed_thickness_attr_name, smoothed_dict)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Layer':
            return Layer
        return super().find_class(module, name)

if __name__ == '__main__':
    import pickle
    Parser.initialize()
    from src.cell.layer.layer import Layer
    with open(r'C:\Users\BardetJ\Downloads\layers_.pkl', 'rb') as f:
        layers = CustomUnpickler(f).load()
    with open(r'C:\Users\BardetJ\Downloads\densities.pkl', 'rb') as f:
        densities = pickle.load(f)
    path_manager = ProcessingPathManager(Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\run\Subject104\Session492'))
    # path_manager.montage.initialize_montage()
    dla = DensityLayerAnalysis(path_manager, densities[0], layers)
    dla.process_layer_densities()
    dla.write_results()

