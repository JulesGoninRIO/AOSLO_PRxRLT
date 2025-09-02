from typing import Dict, List, Tuple
import math
import re
from pathlib import Path
import io
import PIL.Image
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

from src.plotter.plotter import Plotter
from src.configs.parser import Parser
from src.shared.helpers.direction import Direction

class DensityStatisticsPlotter(Plotter):
    def __init__(self, output_path: Path):
        """
        Initialize the DensityStatisticsPlotter.

        :param output_path: The path where the output will be saved.
        :type output_path: Path
        """
        super().__init__(output_path)
        self.step = 0.1

    def gif_layers_to_density(self, base_path: Path):
        """
        Create a GIF from the layers-compared-to-density images, to visualize if/how the layers relate to the density.
        Parser.initialize()
        """
        for direction in Direction:
            frames = []
            for img_path in base_path.glob(f'Subject*/Session*/{Parser.get_density_analysis_dir()}/layers_compared_to_density_{direction.value}.png'):
                frames.append(PIL.Image.open(img_path))

            interpolating_frames = 2   # number of interpolating frames between each frame
            ms_per_frame         = 1000
            ms_per_interp_frame  = 50

            # slightly blend each frame with the next one
            frames_li = sum(([PIL.Image.blend(frames[i], frames[i+1], alpha) for alpha in np.arange(0, 1, 1/(interpolating_frames+1))] for i in range(len(frames)-1)), []) + [frames[-1]]
            durations = ([ms_per_frame] + [ms_per_interp_frame] * interpolating_frames) * (len(frames)-1) + [ms_per_frame]

            # add a white frame at the end
            frames_li.append(PIL.Image.new('RGB', frames_li[0].size, (255, 255, 255)))
            durations.append(1000)

            gif_buffer = io.BytesIO()
            frames_li[0].save(gif_buffer, format='GIF', save_all=True, append_images=frames_li[1:], duration=durations, loop=0)
            gif_buffer.seek(0)
            with open(self.output_path / f'layers_to_density_{direction.value}.gif', 'wb') as f:
                f.write(gif_buffer.read())

    def plot(self, resulting_df: pd.DataFrame):
        """
        Plot various density statistics from the resulting DataFrame.

        :param resulting_df: The DataFrame containing the density statistics.
        :type resulting_df: pd.DataFrame
        """
        by_row_location = resulting_df.groupby(resulting_df.location)
        median_df = by_row_location.median()
        sd_df = by_row_location.std()

        smoothed_column = {}
        for column in median_df:
            # self.plot_density_statistics(column, median_df)
            self.plot_smoothed_density_statistics(column, median_df, smoothed_column)
            self.plot_median_errorbar_statistics(column, median_df, sd_df)

        self.plot_layer_thickness_vs_cone_density(smoothed_column)

    def plot_density_statistics(self, column: str, median_df: pd.DataFrame):
        """
        Plot raw density statistics for a given column.

        :param column: The column name to plot.
        :type column: str
        :param median_df: The DataFrame containing median values.
        :type median_df: pd.DataFrame
        """
        direction = Direction.from_str(column)
        out_name, tick = self._prepare_plot(column, median_df, direction, 'raw')
        plt.plot(median_df.index.values, median_df[column].values)
        plt.savefig(self.output_path / f'raw_{out_name}.png')
        plt.close()

    def plot_smoothed_density_statistics(self, column: str, median_df: pd.DataFrame, smoothed_column: pd.DataFrame):
        """
        Plot smoothed density statistics for a given column.

        :param column: The column name to plot.
        :type column: str
        :param median_df: The DataFrame containing median values.
        :type median_df: pd.DataFrame
        :param smoothed_column: The DataFrame to store smoothed column values.
        :type smoothed_column: pd.DataFrame
        """
        direction = Direction.from_str(column)
        out_name, tick = self._prepare_plot(column, median_df, direction, 'smoothed')
        lowess = sm.nonparametric.lowess(
            list(median_df[column].values), list(median_df.index.values), frac=self.step)
        smoothed_column[column] = lowess
        plt.plot(lowess[:, 0], lowess[:, 1])
        plt.savefig(self.output_path / f'smoothed_{out_name}.png')
        plt.close()

    def plot_median_errorbar_statistics(self, column: str, median_df: pd.DataFrame, sd_df: pd.DataFrame):
        """
        Plot median error bar statistics for a given column.

        :param column: The column name to plot.
        :type column: str
        :param median_df: The DataFrame containing median values.
        :type median_df: pd.DataFrame
        :param sd_df: The DataFrame containing standard deviation values.
        :type sd_df: pd.DataFrame
        """
        direction = Direction.from_str(column)
        out_name, tick = self._prepare_plot(column, median_df, direction, 'Standard Error smoothed')
        try:
            plt.errorbar(
                median_df.index.values, median_df[column].values, sd_df[column].values,
                linestyle='None', marker='.', color='blue', ecolor='orange')
        except ValueError:
            return
        plt.savefig(self.output_path / f'median_errorbar_{out_name}.png')
        plt.close()

    def plot_layer_thickness_vs_cone_density(self, smoothed_column: pd.DataFrame):
        """
        Plot layer thickness vs. cone density for both directions.

        :param smoothed_column: The DataFrame containing smoothed column values.
        :type smoothed_column: pd.DataFrame
        """
        for direction in Direction:
            self._plot_layer_thickness_vs_cone_density(smoothed_column, direction)

    def _prepare_plot(self, column: str, median_df: pd.DataFrame, direction: Direction, plot_type: str) -> Tuple[str, np.ndarray]:
        """
        Prepare the plot for a given column and plot type.

        :param column: The column name to plot.
        :type column: str
        :param median_df: The DataFrame containing median values.
        :type median_df: pd.DataFrame
        :param direction: Direction to be plotted.
        :type direction: Direction
        :param plot_type: The type of plot (e.g., 'raw' or 'smoothed').
        :type plot_type: str
        :return: A tuple containing the output name and tick values.
        :rtype: Tuple[str, np.ndarray]
        """
        if 'densities' in column:
            out_name = column
            ylabel = 'Cone Density [cells/mm²]'
            title = f'Median {plot_type} density for {direction.value}-axis'
        else:
            name = re.sub('_', ' ', column[:-2])
            out_name = re.sub(' ', '_', column)
            ylabel = 'Layer Thickness [mm]'
            title = f'Median {plot_type} {name} thickness for {direction.value}-axis'

        tick = np.arange(math.floor(min(median_df.index.values)), math.ceil(max(median_df.index.values)) + 1, 1)
        plt.xticks(tick, tick, fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel('Retinal Eccentricity [°]', fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.title(title, fontsize=24)
        return out_name, tick

    def _plot_layer_thickness_vs_cone_density(self, smoothed_column: Dict[str, np.ndarray], direction: Direction):
        """
        Plot layer thickness vs. cone density for a given axis.

        :param smoothed_column: The dictionary containing smoothed column values.
        :type smoothed_column: Dict[str, np.ndarray]
        :param direction: Direction to be plotted.
        :type direction: Direction
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.grid(False) # keep only the grid aligned to densities
        lines = []
        for column_name, lowess in smoothed_column.items():
            if f'densities_{direction.value}' in column_name:
                lines.extend(ax2.plot(lowess[:, 0], lowess[:, 1], 'k--', label='cone density'))
            elif f'OS_{direction.value}' in column_name:
                # as OS layer is very thin, we multiply it by 20 to make it
                # visible and meaningfully comparable to other layers/density
                os_values = 20 * lowess[:, 1]
                lines.extend(ax1.plot(lowess[:, 0], os_values, label='OS (×20)', color='darkolivegreen'))
            elif f'_{direction.value}' in column_name:
                lines.extend(ax1.plot(lowess[:, 0], lowess[:, 1], label=f'{column_name[:-2]}'))

        ax1.set_ylabel('Layer thickness [mm] / CV Index', fontsize=14)
        ax2.set_ylabel('Cone density [cones/mm²]', fontsize=14)
        ax1.set_xlabel('Retinal eccentricity [°]', fontsize=14)
        ax1.legend(lines, [line.get_label() for line in lines], loc='upper right', fontsize=14)
        ax1.set_ylim(0, 0.8)
        ax2.set_ylim(0, 300_000)
        ax1.set_xlim(-10, 10)
        ax1.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='both', labelsize=14)

        plt.title(f'Median layer thickness compared to median cone density, {direction.value}-axis', fontsize=18)
        plt.savefig(self.output_path / f'layers_compared_to_densitym_{direction.value}.png')
        plt.close()

    def plot_layer_to_density(self, resulting_df: pd.DataFrame) -> None:
        """
        Plot layer vs density for all layers thicknesses

        :param median_df: the results of the densities and layer thicknesses
        :type median_df: pd.DataFrame
        """
        # maybe add dropna if there are any
        by_row_location = resulting_df.groupby(resulting_df.location)
        median_df = by_row_location.median()
        sd_df = by_row_location.std()

        median_df_truncated = median_df.truncate(before=-10, after=10).drop(-0.0)
        layers = [name for name in median_df.columns if not 'densities' in name]
        for layer in layers:
            direction = Direction.from_str(layer)
            self.plot_density_to_layer(median_df_truncated, layer, direction)
            self.plot_density_to_layer(median_df_truncated, layer, direction, smoothed=True)

    def plot_density_to_layer(self, median_df: pd.DataFrame, layer: str, direction: Direction, smoothed: bool = False) -> None:
        """
        Plot the density versus the layer thickness

        :param median_df: the median results of the densities and layer thicknesses
        :type median_df: pd.DataFrame
        :param layer: the layer name
        :type layer: str
        :param direction: the direction to be plotted.
        :type direction: Direction
        :param smoothed: whether to smooth the plot or not, defaults to False
        :type smoothed: bool, optional
        """

        density = f'densities_{direction.value}'
        if smoothed:
            smoothed_layer = sm.nonparametric.lowess(
                median_df[layer].values, median_df.index.values, frac=0.1)
            smoothed_density = sm.nonparametric.lowess(
                median_df[density].values, median_df.index.values, frac=0.1)
            layer_index = smoothed_layer[:, 0]
            density_index = smoothed_density[:, 0]
            layer_values = smoothed_layer[:, 1]
            density_values = smoothed_density[:, 1]
        else:
            layer_index = median_df.index.values
            density_index = median_df.index.values
            layer_values = median_df[layer].values
            density_values = median_df[density].values

        # plot with the informations
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(layer_index, layer_values, color='b', label=f'{layer} thickness')
        ax2.plot(density_index, density_values, color='r', label='cone density')
        ax1.set_ylabel('Layer Thickness [mm]', fontsize=20)
        ax2.set_ylabel('Cone Density [Cells/mm²]', fontsize=20)
        ax1.set_xlabel('Retinal Eccentricity [°]', fontsize=20)
        ax1.legend(loc='upper left', fontsize=20)
        ax2.legend(loc='upper right', fontsize=20)
        ax1.tick_params(axis='both', labelsize=18)
        ax2.tick_params(axis='both', labelsize=18)
        name = layer[:-2]
        name = re.sub('_', ' ', name)
        out_name = re.sub(' ', '_', layer)
        if smoothed:
            plt.title(
                f'Smoothed median {name} thickness \n compared to median cone density in the {direction.value} direction', fontsize=24)
            plt.savefig(self.output_path / f'{out_name}_smoothed.png')
        else:
            plt.title(
                f'Median {name} thickness \n compared to median cone density in the {direction.value} direction', fontsize=24)
            plt.savefig(self.output_path / f'{out_name}.png')
        plt.tight_layout()
        plt.close()

    def plot_spearmans(self, results_X: pd.DataFrame, results_Y: pd.DataFrame):
        self.plot_violin(
            [
                results_X['RNFL_left'][0],
                results_X['PhotoR+RPE_left'][0],
                results_X['CVI_left'][0],
                results_X['Choroid_left'][0],
                results_X['ONL_left'][0],
                results_X['OS_left'][0],
                results_X['INL+OPL_first_part'][0],
                results_X['INL+OPL_second_part'][0],
                results_X['GCL+IPL_first_part'][0],
                results_X['GCL+IPL_second_part'][0]
            ],
            [
                f'RNFL {results_X['RNFL_left'][1]:.0%}',
                f'PhotoR+RPE {results_X['PhotoR+RPE_left'][1]:.0%}',
                f'CVI {results_X['CVI_left'][1]:.0%}',
                f'Choroid {results_X['Choroid_left'][1]:.0%}',
                f'ONL {results_X['ONL_left'][1]:.0%}',
                f'OS {results_X['OS_left'][1]:.0%}',
                f'INL+OPL \n outer part {results_X['INL+OPL_first_part'][1]:.0%}',
                f'INL+OPL \n central {results_X['INL+OPL_second_part'][1]:.0%}',
                f'GCL+IPL \n outer part {results_X['GCL+IPL_first_part'][1]:.0%}',
                f'GCL+IPL \n central {results_X['GCL+IPL_second_part'][1]:.0%}'
            ],
            'Spearman correlation for Temporal side, p-values < 0.05',
            'spearman_correlation_for_temporal.png'
        )

        self.plot_violin(
            [
                results_X['RNFL_right'][0],
                results_X['PhotoR+RPE_right'][0],
                results_X['CVI_right'][0],
                results_X['Choroid_right'][0],
                results_X['ONL_right'][0],
                results_X['OS_right'][0],
                results_X['INL+OPL_third_part'][0],
                results_X['INL+OPL_fourth_part'][0],
                results_X['GCL+IPL_third_part'][0],
                results_X['GCL+IPL_fourth_part'][0]
            ],
            [
                f'RNFL {results_X['RNFL_right'][1]:.0%}',
                f'PhotoR+RPE {results_X['PhotoR+RPE_right'][1]:.0%}',
                f'CVI {results_X['CVI_right'][1]:.0%}',
                f'Choroid {results_X['Choroid_right'][1]:.0%}',
                f'ONL {results_X['ONL_right'][1]:.0%}',
                f'OS {results_X['OS_right'][1]:.0%}',
                f'INL+OPL \n central {results_X['INL+OPL_third_part'][1]:.0%}',
                f'INL+OPL \n outer {results_X['INL+OPL_fourth_part'][1]:.0%}',
                f'GCL+IPL \n central {results_X['GCL+IPL_third_part'][1]:.0%}',
                f'GCL+IPL \n outer {results_X['GCL+IPL_fourth_part'][1]:.0%}'
            ],
            'Spearman correlation for Nasal side, p-values < 0.05',
            'spearman_correlation_for_nasal.png'
        )

        self.plot_violin(
            [
                results_Y['RNFL_left'][0],
                results_Y['PhotoR+RPE_left'][0],
                results_Y['CVI_left'][0],
                results_Y['Choroid_left'][0],
                results_Y['ONL_left'][0],
                results_Y['OS_left'][0],
                results_Y['INL+OPL_first_part'][0],
                results_Y['INL+OPL_second_part'][0],
                results_Y['GCL+IPL_first_part'][0],
                results_Y['GCL+IPL_second_part'][0]
            ],
            [
                f'RNFL {results_Y['RNFL_left'][1]:.0%}',
                f'PhotoR+RPE {results_Y['PhotoR+RPE_left'][1]:.0%}',
                f'CVI {results_Y['CVI_left'][1]:.0%}',
                f'Choroid {results_Y['Choroid_left'][1]:.0%}',
                f'ONL {results_Y['ONL_left'][1]:.0%}',
                f'OS {results_Y['OS_left'][1]:.0%}',
                f'INL+OPL \n outer part {results_Y['INL+OPL_first_part'][1]:.0%}',
                f'INL+OPL \n central {results_Y['INL+OPL_second_part'][1]:.0%}',
                f'GCL+IPL \n outer part {results_Y['GCL+IPL_first_part'][1]:.0%}',
                f'GCL+IPL \n central {results_Y['GCL+IPL_second_part'][1]:.0%}'
            ],
            'Spearman correlation for Superior side, p-values < 0.05',
            'spearman_correlation_for_superior.png'
        )

        self.plot_violin(
            [
                results_Y['RNFL_right'][0],
                results_Y['PhotoR+RPE_right'][0],
                results_Y['CVI_right'][0],
                results_Y['Choroid_right'][0],
                results_Y['ONL_right'][0],
                results_Y['OS_right'][0],
                results_Y['INL+OPL_third_part'][0],
                results_Y['INL+OPL_fourth_part'][0],
                results_Y['GCL+IPL_third_part'][0],
                results_Y['GCL+IPL_fourth_part'][0]
            ],
            [
                f'RNFL {results_Y['RNFL_right'][1]:.0%}',
                f'PhotoR+RPE {results_Y['PhotoR+RPE_right'][1]:.0%}',
                f'CVI {results_Y['CVI_right'][1]:.0%}',
                f'Choroid {results_Y['Choroid_right'][1]:.0%}',
                f'ONL {results_Y['ONL_right'][1]:.0%}',
                f'OS {results_Y['OS_right'][1]:.0%}',
                f'INL+OPL \n central {results_Y['INL+OPL_third_part'][1]:.0%}',
                f'INL+OPL \n outer {results_Y['INL+OPL_fourth_part'][1]:.0%}',
                f'GCL+IPL \n central {results_Y['GCL+IPL_third_part'][1]:.0%}',
                f'GCL+IPL \n outer {results_Y['GCL+IPL_fourth_part'][1]:.0%}'
            ],
            'Spearman correlation for Inferior side, p-values < 0.05',
            'spearman_correlation_for_inferior.png'
        )

        self.plot_violin(
            [
                results_X['RNFL_left'][0],
                results_X['RNFL_right'][0],
                results_Y['RNFL_left'][0],
                results_Y['RNFL_right'][0]
            ],
            [
                f'RNFL Temporal {results_X['RNFL_left'][1]:.0%}',
                f'RNFL Nasal {results_X['RNFL_right'][1]:.0%}',
                f'RNFL Superior {results_Y['RNFL_left'][1]:.0%}',
                f'RNFL Inferior {results_Y['RNFL_right'][1]:.0%}'
            ],
            'Spearman correlation for RNFL and p-values < 0.05',
            'spearman_correlation_for_RNFL.png'
        )

        self.plot_violin(
            [
                results_X['PhotoR+RPE_left'][0],
                results_X['PhotoR+RPE_right'][0],
                results_Y['PhotoR+RPE_left'][0],
                results_Y['PhotoR+RPE_right'][0]
            ],
            [
                f'PhotoR+RPE Temporal {results_X['PhotoR+RPE_left'][1]:.0%}',
                f'PhotoR+RPE Nasal {results_X['PhotoR+RPE_right'][1]:.0%}',
                f'PhotoR+RPE Superior {results_Y['PhotoR+RPE_left'][1]:.0%}',
                f'PhotoR+RPE Inferior {results_Y['PhotoR+RPE_right'][1]:.0%}'
            ],
            'Spearman correlation for PhotoR+RPE and p-values < 0.05',
            'spearman_correlation_for_PR_RPE.png'
        )

        self.plot_violin(
            [
                results_X['OS_left'][0],
                results_X['OS_right'][0],
                results_Y['OS_left'][0],
                results_Y['OS_right'][0]
            ],
            [
                f'OS Temporal {results_X['OS_left'][1]:.0%}',
                f'OS Nasal {results_X['OS_right'][1]:.0%}',
                f'OS Superior {results_Y['OS_left'][1]:.0%}',
                f'OS Inferior {results_Y['OS_right'][1]:.0%}'
            ],
            'Spearman correlation for Outer Segment and p-values < 0.05',
            'spearman_correlation_for_OS.png'
        )

        self.plot_violin(
            [
                results_X['CVI_left'][0],
                results_X['CVI_right'][0],
                results_Y['CVI_left'][0],
                results_Y['CVI_right'][0]
            ],
            [
                f'CVI Temporal {results_X['CVI_left'][1]:.0%}',
                f'CVI Nasal {results_X['CVI_right'][1]:.0%}',
                f'CVI Superior {results_Y['CVI_left'][1]:.0%}',
                f'CVI Inferior {results_Y['CVI_right'][1]:.0%}'
            ],
            'Spearman correlation for Choroidal \n Vascularity Index and p-values < 0.05',
            'spearman_correlation_for_CVI.png'
        )

        self.plot_violin(
            [
                results_X['CVI_left'][0],
                results_X['CVI_right'][0],
                results_Y['CVI_left'][0],
                results_Y['CVI_right'][0]
            ],
            [
                f'Choroid Temporal {results_X['CVI_left'][1]:.0%}',
                f'Choroid Nasal {results_X['CVI_right'][1]:.0%}',
                f'Choroid Superior {results_Y['CVI_left'][1]:.0%}',
                f'Choroid Inferior {results_Y['CVI_right'][1]:.0%}'
            ],
            'Spearman correlation for Choroid and p-values < 0.05',
            'spearman_correlation_for_CR.png'
        )

        self.plot_violin(
            [
                results_X['ONL_left'][0],
                results_X['ONL_right'][0],
                results_Y['ONL_left'][0],
                results_Y['ONL_right'][0]
            ],
            [
                f'ONL Temporal {results_X['ONL_left'][1]:.0%}',
                f'ONL Nasal {results_X['ONL_right'][1]:.0%}',
                f'ONL Superior {results_Y['ONL_left'][1]:.0%}',
                f'ONL Inferior {results_Y['ONL_right'][1]:.0%}'
            ],
            'Spearman correlation for Outer \n Nuclear layer and p-values < 0.05',
            'spearman_correlation_for_ONL.png'
        )

        self.plot_violin(
            [
                results_X['INL+OPL_first_part'][0],
                results_X['INL+OPL_second_part'][0],
                results_X['INL+OPL_third_part'][0],
                results_X['INL+OPL_fourth_part'][0],
                results_Y['INL+OPL_first_part'][0],
                results_Y['INL+OPL_second_part'][0],
                results_Y['INL+OPL_third_part'][0],
                results_Y['INL+OPL_fourth_part'][0]
            ],
            [
                f'INL+OPL outer \n Temporal {results_X['INL+OPL_first_part'][1]:.0%}',
                f'INL+OPL central \n Temporal {results_X['INL+OPL_second_part'][1]:.0%}',
                f'INL+OPL central \n Nasal {results_X['INL+OPL_third_part'][1]:.0%}',
                f'INL+OPL outer \n Nasal {results_X['INL+OPL_fourth_part'][1]:.0%}',
                f'INL+OPL outer \n Superior {results_Y['INL+OPL_first_part'][1]:.0%}',
                f'INL+OPL central \n Superior {results_Y['INL+OPL_second_part'][1]:.0%}',
                f'INL+OPL central \n Inferior {results_Y['INL+OPL_third_part'][1]:.0%}',
                f'INL+OPL outer \n Inferior {results_Y['INL+OPL_fourth_part'][1]:.0%}'
            ],
            'Spearman correlation for INL+OPL and p-values < 0.05',
            'spearman_correlation_for_INL_OPL.png'
        )

        self.plot_violin(
            [
                results_X['GCL+IPL_first_part'][0],
                results_X['GCL+IPL_second_part'][0],
                results_X['GCL+IPL_third_part'][0],
                results_X['GCL+IPL_fourth_part'][0],
                results_Y['GCL+IPL_first_part'][0],
                results_Y['GCL+IPL_second_part'][0],
                results_Y['GCL+IPL_third_part'][0],
                results_Y['GCL+IPL_fourth_part'][0]
            ],
            [
                f'GCL+IPL outer \n Temporal {results_X['GCL+IPL_first_part'][1]:.0%}',
                f'GCL+IPL central \n Temporal {results_X['GCL+IPL_second_part'][1]:.0%}',
                f'GCL+IPL central \n Nasal {results_X['GCL+IPL_third_part'][1]:.0%}',
                f'GCL+IPL outer \n Nasal {results_X['GCL+IPL_fourth_part'][1]:.0%}',
                f'GCL+IPL outer \n Superior {results_Y['GCL+IPL_first_part'][1]:.0%}',
                f'GCL+IPL central \n Superior {results_Y['GCL+IPL_second_part'][1]:.0%}',
                f'GCL+IPL central \n Inferior {results_Y['GCL+IPL_third_part'][1]:.0%}',
                f'GCL+IPL outer \n Inferior {results_Y['GCL+IPL_fourth_part'][1]:.0%}'
            ],
            'Spearman correlation for GCL+IPL and p-values < 0.05',
            'spearman_correlation_for_GCL_IPL.png'
        )

    def plot_violin(self, data, labels, title, filename):
        fig, ax = plt.subplots()
        try:
            plt.violinplot(data)
        except ValueError:
            plt.close()
            return
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=18, rotation=70)
        ax.set_ylim(-1, 1)
        plt.title(title, fontsize=20)
        plt.ylabel('Spearman correlation', fontsize=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.tight_layout()
        plt.savefig(self.output_path / filename)
        plt.close()

    def plot_median_density_compare_to_median_choroid(self, results_df) -> None:
        """
        Compare the mean density in the center region of the eye to the mean Choroidal
        Thickness and see whether we can find a correlation between those
        """
        plt.scatter(results_df.densities_X_median, results_df.Choroid_X_median)
        plt.ylabel('Layer Thickness [mm]', fontsize=18)
        plt.xlabel('Cone Density [Cells/mm²]', fontsize=18)
        plt.title(
            'Correlation between Median Choroid Thickness \n and Median Densities for each subject for X-axis',
            fontsize=20
        )
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'choroid_X_median.png')
        plt.close()
        plt.scatter(results_df.densities_X_mean, results_df.Choroid_X_mean)
        plt.ylabel('Layer Thickness [mm]', fontsize=18)
        plt.xlabel('Cone Density [Cells/mm²]', fontsize=18)
        plt.title(
            'Correlation between Mean Choroid Thickness \n and Mean Densities for each subject for X-axis',
            fontsize=20
        )

        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'choroid_X_mean.png')
        plt.close()
        plt.scatter(results_df.densities_X_std, results_df.Choroid_X_std)
        plt.ylabel('Layer Thickness [mm]', fontsize=18)
        plt.xlabel('Cone Density [Cells/mm²]', fontsize=18)
        plt.title(
            'Correlation between Standard Deviation Choroid Thickness \n and Standard Deviation Densities for each subject for X-axis',
            fontsize=20
        )

        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'choroid_X_std.png')
        plt.close()

        plt.scatter(results_df.densities_Y_median, results_df.Choroid_Y_median)
        plt.ylabel('Layer Thickness [mm]', fontsize=18)
        plt.xlabel('Cone Density [Cells/mm²]', fontsize=18)
        plt.title(
            'Correlation between Median Choroid Thickness \n and Median Densities for each subject for Y-axis',
            fontsize=20
        )

        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'choroid_Y_median.png')
        plt.close()
        plt.scatter(results_df.densities_Y_mean, results_df.Choroid_Y_mean)
        plt.ylabel('Layer Thickness [mm]', fontsize=18)
        plt.xlabel('Cone Density [Cells/mm²]', fontsize=18)
        plt.title(
            'Correlation between Mean Choroid Thickness \n and Mean Densities for each subject for Y-axis',
            fontsize=20
        )

        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'choroid_Y_mean.png')
        plt.close()
        plt.scatter(results_df.densities_Y_std, results_df.Choroid_Y_std)
        plt.ylabel('Layer Thickness [mm]', fontsize=18)
        plt.xlabel('Cone Density [Cells/mm²]', fontsize=18)
        plt.title(
            'Correlation between Standard Deviation Choroid Thickness \n and Standard Deviation Densities for each subject for Y-axis',
            fontsize=20
        )
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'choroid_Y_std.png')
        plt.close()

    def plot_X_vs_Y(self, median_df: pd.DataFrame, sd_df: pd.DataFrame) -> None:
        """
        Plot the X-axis versus the y-one for densities and layer thicknesses

        :param median_df: the DataFrame containing the medians of the values
        :type median_df: pd.DataFrame
        """

        # go through column to plot either densities or layer thicknesses
        seen_cols = set()
        for column in median_df:
            if column == 'location':
                continue
            col_name = column[:-2]
            if col_name in seen_cols:
                continue
            seen_cols.add(col_name)
            ttl_name = col_name.replace('_', ' ')
            # axis = 'x' if '_X' in column else 'y'
            # direction = Direction.from_str(column)
            tick = np.arange(
                math.floor(min(median_df.index.values)),
                math.ceil(max(median_df.index.values))+1,
                1
            )
            plt.xticks(tick, tick, fontsize=20)
            plt.tick_params(axis='both', labelsize=18)
            plt.xlabel('Retinal Eccentricity [°]', fontsize=20)
            plt.ylabel('Cone Density [cells/mm²]' if 'density' in column else 'Layer Thickness [mm]', fontsize=20)
            plt.title(f'Median {ttl_name} comparison between axis', fontsize=24)
            for direction in Direction:
                lowess = sm.nonparametric.lowess(
                    list(median_df[f'{col_name}_{direction.value}'].values),
                    list(median_df.index.values),
                    frac=0.1
                )
                plt.plot(lowess[:, 0], lowess[:, 1], label=f'{direction.value}-axis')
            plt.legend(fontsize=20)
            plt.savefig(self.output_path / f'{col_name}_X_vs_Y_comparison.png')
            plt.close()

        # save an excel file
        writer = pd.ExcelWriter(self.output_path / 'X_vs_Y_comparison.xlsx', engine = 'xlsxwriter')
        seen_cols = set()
        for column in median_df:
            if column == 'location':
                continue
            col_name = column[:-2]
            if col_name in seen_cols:
                continue
            seen_cols.add(col_name)
            out_csv_df = pd.DataFrame()
            for direction in Direction:
                out_csv_df[direction.value] = median_df[column].astype('str') + ' (' + sd_df[column].astype('str') + ')'
            out_csv_df.to_excel(writer, sheet_name = column)
            writer.sheets[column].set_column(1, 3, 50)
        writer.close()

