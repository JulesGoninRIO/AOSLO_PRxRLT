import pickle
import numpy as np
from typing import Tuple, Dict, List, Callable

from src.shared.helpers.direction import Direction
from src.configs.parser import Parser
from src.plotter.layer_plotter import LayerPlotter
from src.cell.layer.layer import Layer
from src.cell.analysis.constants import MM_PER_DEGREE, PIXELS_PER_DEGREE
from src.cell.processing_path_manager import ProcessingPathManager

class LayerThicknessCalculator:
    """
    Calculate the thickness of retinal layers.

    :param path_manager: Manager for processing paths and directories
    :type path_manager: ProcessingPathManager
    :param layers: Dictionary mapping layer names to Layer objects
    :type layers: Dict[str, Layer]
    :param spacing: Dictionary containing spacing values for x, y, z dimensions in mm
    :type spacing: Dict[str, float]
    :param center: Peak location coordinates as (y, x) indices
    :type center: Tuple[int, int]

    :ivar layers: Dictionary of layer objects
    :vartype layers: Dict[str, Layer]
    :ivar growing_factor: Function to calculate growing factor
    :vartype growing_factor: Callable[[float], float]
    :ivar step: Step size for calculations
    :vartype step: float
    :ivar round_step: Rounded step size
    :vartype round_step: int
    :ivar spacing: Spacing values for dimensions
    :vartype spacing: Dict[str, float]
    :ivar center: Peak location coordinates
    :vartype center: Tuple[int, int]
    :ivar plotter: Plotter object for visualization
    :vartype plotter: LayerPlotter
    :ivar triangle_plot: Storage for triangle plot coordinates
    :vartype triangle_plot: Dict[str, List[Tuple[int, int]]]
    """
    def __init__(self, path_manager: ProcessingPathManager, layers: Dict[str, Layer], 
                 spacing: Dict[str, float], center: Tuple[int, int]) -> None:
        self.layers: Dict[str, Layer] = layers
        try:
            growing_factor = Parser.get_density_growing_factor()
        except AttributeError:
            Parser.initialize()
            growing_factor = Parser.get_density_growing_factor()
        self.growing_factor: Callable[[float], float] = lambda x: growing_factor[0]*x + growing_factor[1]
        self.step: float = 0.1
        self.round_step: int = round(self.step*10)
        self.spacing: Dict[str, float] = spacing
        self.center: Tuple[int, int] = center
        self.plotter = LayerPlotter(path_manager.density.layer_path, self.spacing)
        self.triangle_plot: Dict[str, List[Tuple[int, int]]] = {}

    def get_layer_thicknesses(self) -> Dict[str, Layer]:
        """
        Calculate thicknesses for all layers.

        Processes each layer in both x and y directions and generates
        3D triangle plots for visualization.

        :return: Updated dictionary of layers with calculated thicknesses
        :rtype: Dict[str, Layer]
        """
        for layer_name, layer in self.layers.items():
            if layer.is_empty():
                continue
            self.triangle_plot = {}
            for direction in Direction:
                self.process_layer(layer_name, direction)
            try:
                self.plotter.plot_3d_triangle(self.triangle_plot, layer, self.center)
            except KeyError:
                continue
            layer.calculate_mean_thickness_per_distance()
            layer.fill_gaps(self.step, self.round_step)
        return self.layers

    def process_layer(self, layer_name: str, direction: Direction) -> None:
        """
        Process a single layer in a specified direction.

        :param layer_name: Name of the layer to process
        :type layer_name: str
        :param direction: Direction enum value for processing orientation
        :type direction: Direction
        """

        # print ("DEBUG: center:", self.center)
        for bscan_idx, bscan in enumerate(self.layers[layer_name].padded_thickness_values):
            for ascan_idx, thickness in enumerate(bscan):
                if thickness is not None and thickness > 0:
                    # if bscan_idx == 47 and layer_name == "OS":
                    #     print(bscan_idx, ascan_idx)
                    #     print(thickness)
                    self.process_thickness(bscan_idx, ascan_idx, thickness, layer_name, direction)

    def process_thickness(self, bscan_idx: int, ascan_idx: int, thickness: float, 
                         layer_name: str, direction: Direction) -> None:
        """
        Process the thickness at a specific point in the layer.

        :param bscan_idx: Y index from Superior to Inferior
        :type bscan_idx: int
        :param ascan_idx: X index from Temporal to Nasal
        :type ascan_idx: int
        :param thickness: Thickness value at the specified point
        :type thickness: float
        :param layer_name: Name of the layer being processed
        :type layer_name: str
        :param direction: Direction enum value for processing orientation
        :type direction: Direction
        """
        # compute the eccentricity of (bscan_idx, ascan_idx) with respect to the center
        # bscan_idx is y index from Superior to Inferior, ascan_idx is x index from Temporal to Nasal

        # differences in pixels [pixel scale of mosaic]
        diff_x_px = round((ascan_idx - self.center[1]) * self.spacing['x']  / MM_PER_DEGREE * PIXELS_PER_DEGREE)
        diff_y_px = round((bscan_idx - self.center[0]) * self.spacing['y']  / MM_PER_DEGREE * PIXELS_PER_DEGREE)
        
        # only consider point if inside the triangle region
        if (abs(diff_x_px) < self.growing_factor(abs(diff_y_px)) and direction.is_Y) \
        or (abs(diff_y_px) < self.growing_factor(abs(diff_x_px)) and direction.is_X):
            # back to degrees
            distance = round(np.sqrt(diff_x_px**2 + diff_y_px**2) / PIXELS_PER_DEGREE, self.round_step)
            if (diff_y_px < 0 and direction.is_Y) or (diff_x_px < 0 and direction.is_X):
                distance *= -1
            attribute_name = f"thickness_per_distance_{direction.value}"

            current_dict: Dict = getattr(self.layers[layer_name], attribute_name, {})
            current_dict.setdefault(distance, []).append(thickness)


            setattr(self.layers[layer_name], attribute_name, current_dict)
            self.triangle_plot.setdefault(layer_name, []).append((bscan_idx, ascan_idx))

if __name__ == "__main__":
    import pickle
    Parser.initialize()
    with open(r"C:\Users\BardetJ\Downloads\layers.pkl", "rb") as f:
        layers = pickle.load(f)
    dla = LayerThicknessCalculator(layers)
    dla.get_layer_thicknesses()
    with open(r"C:\Users\BardetJ\Downloads\layers_.pkl", "wb") as f:
        pickle.dump(layers, f)
    print("oe")

