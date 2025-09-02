from typing import Dict
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from src.plotter.plotter import Plotter
from src.shared.helpers.direction import Direction
from src.cell.layer.layer import Layer
from src.cell.analysis.density import Density

import matplotx

class DensityLayerPlotter(Plotter):
    def __init__(self, output_path: Path):
        """
        Initialize the DensityLayerPlotter.

        :param output_path: The path where the output will be saved.
        :type output_path: Path
        """
        super().__init__(output_path)

    def plot(self):
        """
        Placeholder method for plotting.
        """
        pass

    def plot_thicknesses_to_densities(
    self,
    layers: Dict[str, Layer],
    densities: Density,
    direction: Direction,
    subject_id = None):
        """
        Plot all layer thicknesses compared to densities in one plot.

        :param layers: Dictionary of layers with their thickness data.
        :type layers: Dict[str, Layer]
        :param densities: Densities with their data.
        :type densities: Density
        :param direction: The direction considered (X, Y)
        :type direction: Direction
        """
        with plt.style.context(matplotx.styles.dufte):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.grid(False) # keep only the grid aligned to densities
            ax2.grid(False)

            # ---- ADD BACKGROUND GRADIENT FOR RETINAL REGIONS ----
            # Get the background color from current style
            fig_temp, ax_temp = plt.subplots()
            bg_color = ax_temp.get_facecolor()
            plt.close(fig_temp)
            from matplotlib.colors import to_rgb
            bg_rgb = to_rgb(bg_color) if isinstance(bg_color, str) else bg_color[:3]
            gradient_colors = [
                (1, 1, 1),  # white
                tuple(0.8 + 0.2 * x for x in bg_rgb),  # 80% white + 20% bg
                tuple(0.6 + 0.4 * x for x in bg_rgb),  # 60% white + 40% bg
                tuple(0.4 + 0.6 * x for x in bg_rgb),  # 40% white + 60% bg
                tuple(0.2 + 0.8 * x for x in bg_rgb),  # 20% white + 80% bg
                bg_rgb  # full background color
            ]
            retinal_regions = [
                {'name': 'Perifovea', 'start': 1.25, 'end': np.inf, 'color': gradient_colors[5]},
                {'name': 'Parafovea', 'start': 0.75, 'end': 1.25, 'color': gradient_colors[3]},
                {'name': 'Fovea', 'start': 0.25, 'end': 0.75, 'color': gradient_colors[2]},
                {'name': 'Faz', 'start': 0.175, 'end': 0.25, 'color': gradient_colors[1]},
                {'name': 'Foveola', 'start': -0.175, 'end': 0.175, 'color': gradient_colors[0]},
                {'name': 'Faz', 'start': -0.25, 'end': -0.175, 'color': gradient_colors[1]},
                {'name': 'Fovea', 'start': -0.75, 'end': -0.25, 'color': gradient_colors[2]},
                {'name': 'Parafovea', 'start': -1.25, 'end': -0.75, 'color': gradient_colors[3]},
                {'name': 'Perifovea', 'start': -np.inf, 'end': -1.25, 'color': gradient_colors[5]}
            ]
            for region in retinal_regions:
                ax1.axvspan(region['start'], region['end'], facecolor=region['color'], alpha=1.0, zorder=0)
            # ---- END BACKGROUND GRADIENT ----

            # Use a colorblind friendly discrete colormap for layer plotting
            discrete_cmap = plt.get_cmap("tab10")

            lines = []
            for i, (layer_name, layer) in enumerate(layers.items()):
                smoothed_thickness_attr_name = f'smoothed_thickness_per_distance_{direction.value}'
                layer_value = getattr(layer, smoothed_thickness_attr_name)
                if layer_name == 'OS':
                    # as OS layer is very thin, we multiply it by 20 to make it visible
                    # and meaningfully comparable to other layers/density
                    os_values = 20 * np.array(list(layer_value.values()))
                    lines.extend(ax1.plot(list(layer_value.keys()), os_values, label='OS (×20)', color=discrete_cmap(i)))
                elif ("density" in layer_name.lower()):
                    # skip density layers, as they are plotted separately
                    continue
                else:
                    lines.extend(ax1.plot(list(layer_value.keys()), list(layer_value.values()), label=layer_name, color=discrete_cmap(i)))
            density = getattr(densities, f'{direction.value}_fitted')
            density = getattr(densities, f'{direction.value}_smoothed')
            lines.extend(ax2.plot(density.keys(), density.values(), 'k--', label='Cone density'))

            ax1.set_ylabel('Layer thickness [mm] / CV Index', fontsize=14)
            ax2.set_ylabel('Cone density [cones/mm²]', fontsize=14)
            ax1.set_xlabel('Retinal eccentricity [°]', fontsize=14)

            ax1.legend(lines, [line.get_label() for line in lines], loc='upper right', fontsize=14)
            ax1.set_ylim(0, 0.8)
            # ax2.set_ylim(0, 300_000)
            ax1.tick_params(axis='both', labelsize=14)
            # ax2.tick_params(axis='both', labelsize=14)
            plt.title(
                f'Thicknesses/CVIs compared, {direction.value}-axis, Subject {subject_id}',
                fontsize=18
            )
            plt.xlim(-6, 6)
            plt.tight_layout()
            plt.savefig(self.output_path / f'layers_compared_to_density_{direction.value}.png')
            plt.close()

    def plot_2d_thickness_vs_density(
        self,
        layers: Dict[str, Layer],
        densities: Density,
        layer_name: str,
        direction: Direction):
        """
        Plot 2D thickness vs. density for a specific layer.

        :param layers: Dictionary of layers with their thickness data.
        :type layers: Dict[str, Layer]
        :param densities: Densities with their data.
        :type densities: Density
        :param layer_name: The name of the layer to analyze.
        :type layer_name: str
        :param direction: The direction to analyze (e.g., 'horizontal' or 'vertical').
        :type direction: Direction
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.grid(False) # keep only the grid aligned to densities
        thicknss = getattr(layers[layer_name], f'smoothed_thickness_per_distance_{direction.value}')
        line1, = ax1.plot(thicknss.keys(), thicknss.values(), label=(f'{layer_name} thickness' if layer_name != 'CVI' else 'CVI'))
        density = getattr(densities, f'{direction.value}_smoothed')
        line2, = ax2.plot(density.keys(), density.values(), 'k--', label='Cone density')
        ax1.set_ylabel('Layer thickness [mm]' if layer_name != 'CVI' else 'Choroidal Vascularity Index', fontsize=14)
        ax2.set_ylabel('Cone density [cones/mm²]', fontsize=14)
        ax1.set_xlabel('Retinal eccentricity [°]', fontsize=14)
        
        ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='best', fontsize=14)

        ax1.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='both', labelsize=14)
        name = re.sub('_', ' ', layer_name)
        out_name = re.sub(' ', '_', layer_name)
        plt.title(f'{name} compared to cone density, {direction.value}-axis', fontsize=18)
        plt.savefig(self.output_path / f'{direction.value}_{out_name}.png')
        plt.close()
