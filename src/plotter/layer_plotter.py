import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from typing import Dict, List, Tuple
from pathlib import Path

from src.cell.layer.layer import Layer
from src.cell.analysis.constants import MM_PER_DEGREE
from src.plotter.plotter import Plotter

class LayerPlotter(Plotter):
    def __init__(self, output_path: Path, spacing: Dict[str, float]):
        """
        Initialize the LayerPlotter.

        :param output_path: The path where the output will be saved.
        :type output_path: Path
        """
        super().__init__(output_path)
        self.spacing = spacing


    def plot(
        self,
        layer_name: str,
        layer: Layer,
        grid: Tuple[int, int],
        numbers: Tuple[int, int],
        degrees: Tuple[int, int]):
        """
        Create a 3D plot for the given layer.

        :param layer_name: The name of the layer to plot.
        :type layer_name: str
        :param layer: The layer data to plot.
        :type layer: Layer
        :param grid: The grid dimensions for the plot.
        :type grid: Tuple[int, int]
        :param numbers: The tick numbers for the x and y axes.
        :type numbers: Tuple[int, int]
        :param degrees: The tick labels for the x and y axes.
        :type degrees: Tuple[int, int]
        """
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.tick_params(labelsize=12)

        X, Y = np.meshgrid(grid[0], grid[1])
        surf = ax.plot_surface(X, Y, layer, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, alpha=0.2)
        # ax.tick_params(labelsize=14)
        ax.set_xticks(numbers[0])
        ax.set_xticklabels(degrees[0], fontdict={'fontsize': 12})
        ax.set_yticks(numbers[1])
        ax.set_yticklabels(degrees[1], fontdict={'fontsize': 12})
        ax.set_xlabel('Temporal-Nasal [°]', fontsize=14, labelpad=10)
        ax.set_ylabel('Inferior-Superior [°]', fontsize=14, labelpad=10)
        ax.invert_yaxis()  # to match the mosaic downwards y-axis
        ax.set_zlabel('Layer Thickness [mm]', fontsize=14, labelpad=10)
        ax.set_title(
            f'3D representation of the {layer_name} layer thickness',
            fontsize=14
        )
        plt.savefig(self.output_path / f'{layer_name}.png')
        plt.close()

    def plot_3d_triangle(
        self,
        triangle: Dict[str, List[int]],
        layer: Layer,
        peak: Tuple[int, int]) -> None:
        """
        Create a 3D plot for the given triangular region of the layer.

        :param triangle: The dictionary containing the triangular region indices.
        :type triangle: Dict[str, List[int]]
        :param layer: The layer data to plot.
        :type layer: Layer
        :param peak: The peak coordinates in the layer.
        :type peak: Tuple[int, int]
        :param x: Whether to plot along the x-axis. Defaults to True.
        :type x: bool
        """
        layer_v = layer.padded_thickness_values
        bin_map = np.zeros_like(layer_v)
        # for each (bscan_idx, ascan_idx) in the triangle, set non-zero value 
        bin_map[*zip(*triangle[layer.name])] = np.max(layer_v) / 10

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.tick_params(labelsize=12)

        X = (np.arange(0, bin_map.shape[1], 1) - peak[1]) * self.spacing['x'] / MM_PER_DEGREE
        Y = (np.arange(0, bin_map.shape[0], 1) - peak[0]) * self.spacing['y'] / MM_PER_DEGREE
        X, Y = np.meshgrid(X, Y)
        Z = bin_map
        surf = ax.plot_surface(X, Y, Z, cmap=cm.PiYG, linewidth=0, antialiased=False, alpha=0.2)
        Z = layer_v
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.2)
        # ax.tick_params(labelsize=14)
        ax.set_xlabel('Temporal-Nasal [°]', fontsize=14, labelpad=10)
        ax.set_ylabel('Inferior-Superior [°]', fontsize=14, labelpad=10)
        ax.invert_yaxis() # to match the mosaic downwards y-axis
        ax.set_zlabel('Layer Thickness [mm]', fontsize=14, labelpad=10)
        ax.set_xticks(np.arange(-10, 11, 2))
        ax.set_yticks(np.arange(-10, 11, 2))
        ax.set_title(f'3D representation of the {layer.name} layer thickness', fontsize=14)
        plt.savefig(self.output_path / f'triangle_{layer.name}.png')
        plt.close()