from typing import List, Dict, Tuple
import csv
import pickle
import numpy as np

from src.shared.helpers.direction import Direction
from src.cell.analysis.density import Density
from src.cell.analysis.constants import RESULT_NAME
from src.cell.layer.layer import Layer
from src.cell.processing_path_manager import ProcessingPathManager
from src.configs.parser import Parser


class ResultWriter:
    """
    A class to write density results to a CSV file.

    Attributes
    ----------
    path_manager : PathManager
        An instance of PathManager to manage file paths.
    layers_to_avoid : list
        A list of layers to avoid in the results.
    step : int
        The step size for processing.
    round_step : int
        The rounding step size for processing.

    Methods
    -------
    write_density_results(densities: Density, layers: Dict[str, Layer] = None) -> None:
        Writes the density results to a CSV file.
    _write_layered_density_results(writer, layers, densities):
        Writes the layered density results to the CSV file.
    """
    def __init__(
        self,
        path_manager: ProcessingPathManager,
        layers_to_avoid: List[str],
        step: int,
        round_step: float):
        """
        Initializes the ResultWriter class with the given parameters.

        :param path_manager: An instance of PathManager to manage file paths.
        :type path_manager: PathManager
        :param layers_to_avoid: A list of layers to avoid in the results.
        :type layers_to_avoid: List[str]
        :param step: The step size for processing.
        :type step: int
        :param round_step: The rounding step size for processing.
        :type round_step: int
        """
        self.path_manager = path_manager
        self.layers_to_avoid = layers_to_avoid
        self.step = step
        self.round_step = round_step

    def write_density_results(self, densities: Density, layers: Dict[str, Layer] | None = None) -> None:
        """
        Write the density results to a CSV file.

        This method writes the density results to a CSV file. If layers are provided, it writes
        layered density results; otherwise, it writes only the densities.

        :param densities: The density data to write.
        :type densities: Density
        :param layers: The layers to include in the results, defaults to None.
        :type layers: Dict[str, Layer], optional
        :return: None
        """
        self.densities_x = densities.X_fitted
        self.densities_y = densities.Y_fitted
        # self.densities_x = densities.X_smoothed
        # self.densities_y = densities.Y_smoothed
        result_path = self.path_manager.density.path / RESULT_NAME
        
        with open(result_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',', dialect='excel')
            writer.writerow(["sep=,"])

            if layers is None:
                self.write_only_densities(writer)
            else:
                self._write_layered_density_results(writer, layers)

    def _write_layered_density_results(self, writer: csv.writer, layers: Dict[str, Layer]) -> None:
        """
        Write the layered density results to the CSV file.

        This method writes the layered density results to the CSV file, including layer names and ranges.

        :param writer: The CSV writer object.
        :type writer: csv.writer
        :param layers: The layers to include in the results.
        :type layers: Dict[str, Layer]
        :return: None
        """
        layer_names, layer_range = self._prepare_layer_info(layers)
        self._write_header(writer, layer_names)
        self._write_density_data(writer, layers, layer_names, layer_range)

    def _prepare_layer_info(self, layers: Dict[str, Layer]) -> Tuple[List[str], List[float]]:
        """
        Prepare layer information for writing density results.

        This method prepares the layer names and range for the given layers, excluding the layers to avoid.

        :param layers: A dictionary of layers to process.
        :type layers: Dict[str, Layer]
        :return: A tuple containing the list of layer names and the layer range.
        :rtype: Tuple[List[str], List[float]]
        """
        layer_range = [float('inf'), float('-inf')]
        layer_names = []
        for layer_name, layer in layers.items():
            if layer_name not in self.layers_to_avoid:
                layer_names.append(layer_name)
                for direction in Direction:
                    layer_range[0] = min(layer_range[0], min(getattr(layer, f'mean_thickness_per_distance_{direction.value}').keys()))
                    layer_range[1] = max(layer_range[1], max(getattr(layer, f'mean_thickness_per_distance_{direction.value}').keys()))
        return layer_names, layer_range

    def _write_header(self, writer: csv.writer, layer_names: List[str]) -> None:
        """
        Write the header row for the density results CSV file.

        This method writes the header row for the CSV file, including layer names for both x and y directions.

        :param writer: The CSV writer object.
        :type writer: csv.writer
        :param layer_names: The list of layer names to include in the header.
        :type layer_names: List[str]
        :return: None
        """
        layer_names_x = [f"{name}_X" for name in layer_names]
        layer_names_y = [f"{name}_Y" for name in layer_names]
        row_names = ["location", "densities_X"] + layer_names_x + ["densities_Y"] + layer_names_y
        writer.writerow(row_names)

    def _write_density_data(
        self,
        writer: csv.writer,
        layers: Dict[str, Layer],
        layer_names: List[str],
        layer_range: List[int]) -> None:
        """
        Write the density data to the CSV file.

        This method writes the density data to the CSV file, including densities and layer values for each step.

        :param writer: The CSV writer object.
        :type writer: csv.writer
        :param layers: The layers to include in the results.
        :type layers: Dict[str, Layer]
        :param layer_names: The list of layer names to include in the results.
        :type layer_names: List[str]
        :param layer_range: The range of layers to include in the results.
        :type layer_range: List[int]
        :return: None
        """
        key_range = self.calculate_key_range(layer_range)
        none_if_zero = lambda x: x if x != 0 and x != np.nan else None
        for ecc in np.arange(key_range[0], key_range[1]+self.step, self.step):
            ecc = round(ecc, self.round_step)
            layers_values_x = [layers.get(name, {}).mean_thickness_per_distance_X.get(ecc, None) for name in layer_names]
            layers_values_y = [layers.get(name, {}).mean_thickness_per_distance_Y.get(ecc, None) for name in layer_names]
            densities_x_value = none_if_zero(self.densities_x.get(ecc, None))
            densities_y_value = none_if_zero(self.densities_y.get(ecc, None))
            row_values = [ecc, densities_x_value] + layers_values_x + [densities_y_value] + layers_values_y
            writer.writerow(row_values)

    def calculate_key_range(self, layer_range: List[int]) -> List[int]:
        """
        Calculate the key range for density data.

        This method calculates the key range for the given density data and layer range.

        :param densities: The density data to process.
        :type densities: Density
        :param layer_range: The range of layers to include in the results.
        :type layer_range: List[int]
        :return: The calculated key range.
        :rtype: List[int]
        """
        try:
            return [
                min(np.min(list(self.densities_x.keys())), np.min(list(self.densities_y.keys())), layer_range[0]),
                max(np.max(list(self.densities_x.keys())), np.max(list(self.densities_y.keys())), layer_range[1])
            ]
        except ValueError:
            return layer_range

    def write_only_densities(self, writer: csv.writer) -> None:
        """
        Write only the density data to the CSV file.

        This method writes only the density data to the CSV file, without any layer information.

        :param writer: The CSV writer object.
        :type writer: csv.writer
        :param densities: The density data to write.
        :type densities: Density
        :return: None
        """
        row_names = ["location", "densities_X", "densities_Y"]
        writer.writerow(row_names)
        key_range = self.calculate_key_range([np.inf, -np.inf])
        none_if_zero = lambda x: x if x != 0 and x != np.nan else None
        for i in np.arange(key_range[0], key_range[1], self.step):
            i = round(i, self.round_step)
            writer.writerow([i, none_if_zero(self.densities_x.get(i, None)), none_if_zero(self.densities_y.get(i, None))])

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Layer':
            return Layer
        return super().find_class(module, name)

if __name__ == "__main__":
    from pathlib import Path
    import pickle
    Parser.initialize()
    from src.cell.layer.layer import Layer
    rw = ResultWriter(
        ProcessingPathManager(Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject104\Session492')),
        # density_analysis_path=Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject104\Session492\density_analysis'),
        # checkpoint_path=Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject104\Session492\checkpoint'),
        # dir_to_process=Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject104\Session492'),
        # layers_to_avoid=["IRF", "PED", "SRF", "UNKNOWN", "CVI"],
        step=0.1,
        round_step=1
    )
    with open(r"C:\Users\BardetJ\Downloads\layers_.pkl", "rb") as f:
        layers = CustomUnpickler(f).load()
    with open(r"C:\Users\BardetJ\Downloads\densities.pkl", "rb") as f:
        densities = pickle.load(f)
    rw.write_density_results(densities[0], layers)
    print("Done!")
