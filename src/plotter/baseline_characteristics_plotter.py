from typing import Dict, List, Callable
from pathlib import Path
import os
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd

from src.plotter.plotter import Plotter
from src.shared.helpers.direction import Direction

class BaselineCharacteristicsPlotter(Plotter):
    """
    A class to plot baseline characteristics such as age, axial length, and spherical equivalence.

    :param output_path: The path where the plots will be saved.
    :type output_path: Path
    :param step: The step size for smoothing.
    :type step: float
    :param resolution: The resolution of the plots.
    :type resolution: int
    :param age_bins: The bins for age groups.
    :type age_bins: List[int]
    :param axial_length_bins: The bins for axial length groups.
    :type axial_length_bins: List[int]
    :param spherical_equiv_bins: The bins for spherical equivalence groups.
    :type spherical_equiv_bins: List[Tuple[int, int]]
    :param visit_dates_bins: The bins for visit dates.
    :type visit_dates_bins: List[datetime]
    :param min_ecc: The minimum eccentricity.
    :type min_ecc: int
    :param max_ecc: The maximum eccentricity.
    :type max_ecc: int
    :param eccentricity_bins: The bins for eccentricity.
    :type eccentricity_bins: np.array
    """

    def __init__(
        self,
        output_path: Path,
        min_ecc: int,
        max_ecc: int,
        eccentricity_bins: np.ndarray):
        super().__init__(output_path)
        if not self.output_path.exists():
            self.output_path.mkdir()
        self.min_ecc = min_ecc
        self.max_ecc = max_ecc
        self.eccentricity_bins = eccentricity_bins

    def plot_errorbars(
        self,
        age_median_dfs: List[pd.DataFrame],
        age_std_dfs: List[pd.DataFrame],
        age_range_dfs: List[pd.DataFrame],
        label_callback: Callable[[int, int], str]):
        """
        Plot the error bars for cone densities across different age groups.
        """
        for direction in Direction:
            tick = np.arange(math.floor(self.min_ecc), math.ceil(self.max_ecc) + 1, 1)
            plt.xticks(tick, tick, fontsize=20)
            plt.tick_params(axis='both', labelsize=18)
            plt.xlabel("Retinal eccentricity [°]", fontsize=20)
            plt.ylabel("Cone density [cells/mm^2]", fontsize=20)
            plt.title(f"Median density for {direction.value}-axis", fontsize=24)

            colors = ["green", "red", "purple", "black"]
            for i, (median_df, std_df, range_df) in enumerate(zip(age_median_dfs, age_std_dfs, age_range_dfs)):
                plt.errorbar(
                    self.eccentricity_bins,
                    median_df[f'densities_{direction.value}'],
                    yerr=std_df[f'densities_{direction.value}'],
                    label=label_callback(i, len(range_df)),
                    color=colors[i],
                    ecolor=colors[i]
                )
            plt.legend(fontsize=12)
            plt.xlim(self.min_ecc, self.max_ecc)
            plt.savefig(self.output_path / f"age_density_{direction.value}_errorbar.png")
            plt.close()

    def _name_to_ylabel(self, name: str) -> str:
        if "densities" in name:
            return "Cone Density [cells/mm²]"
        elif "CVI" in name:
            return "Choroidal Vascularity Index"
        else:
            return "Layer Thickness [mm]"
    
    def _name_to_title(self, name: str, plural: bool = False) -> str:
        if "densities" in name:
            return "densit" + ("ies" if plural else "y")
        elif "CVI" in name:
            return "CVI" + ("s" if plural else "")
        else:
            return f"{name.split("_")[0]} layer thickness" + ("es" if plural else "")

    def plot_characteristics(
            self,
            characteristic: str,
            median_dfs: list[pd.DataFrame],
            range_dfs: list[pd.DataFrame],
            label_callback: Callable[[int, int], str]
        ):
        
        for column in median_dfs[0].columns:
            if column == "location":
                continue

            direction = Direction.from_str(column)
            out_name = column.replace(" ", "_")
            tick = np.arange(math.floor(self.min_ecc), math.ceil(self.max_ecc) + 1, 1)
            plt.xticks(tick, tick, fontsize=20)
            plt.tick_params(axis='both', labelsize=18)
            plt.xlabel("Retinal eccentricity [°]", fontsize=20)
            plt.ylabel(self._name_to_ylabel(column), fontsize=20)
            plt.title(f"Median {self._name_to_title(column)} for {direction.value}-axis", fontsize=24)

            # don't smoothen again densities, they've already been smoothed
            prepare_data = lambda x, c: np.where(c.isna(), np.nan, sm.nonparametric.lowess(c.to_numpy(), x, frac=0.1, xvals=x)) if 'densities' not in c.name else c.to_numpy()

            colors = ["green", "red", "purple", "black"]
            for i, (median_df, range_df) in enumerate(zip(median_dfs, range_dfs)):
                plt.plot(
                    self.eccentricity_bins,
                    prepare_data(self.eccentricity_bins, median_df[column]),
                    color=colors[i],
                    label=label_callback(i, len(range_df))
                )
            
            plt.legend(fontsize=12)
            plt.xlim(self.min_ecc, self.max_ecc)
            plt.savefig(self.output_path / f"{characteristic}_{out_name}.png")
            plt.close()

    def plot_subject_against_subject(self, subjects_dirs: Dict[str, pd.DataFrame]):
        """
        Plot subject data against each other.

        This method plots the layer thicknesses and densities for different subjects.

        :param subjects_dirs: Dictionary containing DataFrames for each subject.
        :type subjects_dirs: Dict[str, pd.DataFrame]
        """
        # sort the dictionary
        subjects_dirs = collections.OrderedDict(sorted(subjects_dirs.items(), key=lambda x: int(x[0].lstrip('Subject'))))
        for column in subjects_dirs[list(subjects_dirs.keys())[0]][-1].columns:
            if column == "location":
                continue
            fig = plt.figure()
            ax = fig.add_subplot(111)
            direction = Direction.from_str(column)
            out_name = column.replace(" ", "_")
            tick = np.arange(math.floor(self.min_ecc), math.ceil(self.max_ecc) + 1, 1)
            plt.xticks(tick, tick, fontsize=20)
            plt.tick_params(axis='both', labelsize=18)
            plt.xlabel("Retinal eccentricity [°]", fontsize=20)
            plt.ylabel(self._name_to_ylabel(column), fontsize=20)
            plt.title(f"All {self._name_to_title(column, plural=True)} for {direction.value}-axis", fontsize=24)

            prepare_data = lambda x, c: np.where(c.isna(), np.nan, sm.nonparametric.lowess(c.to_numpy(), x, frac=0.1, xvals=x)) if 'densities' not in c.name else c.to_numpy()
            for key, infos in subjects_dirs.items():
                # series = infos[-1][column]
                # if 'densities' in series.name:
                #     densities = series.to_numpy()
                #     if np.any(densities > 400_000):
                #         print(f'{key} has densities > 400_000 for {column}_{direction.value}')
                try:
                    plt.plot(
                        self.eccentricity_bins,
                        prepare_data(self.eccentricity_bins, infos[-1][column]),
                        label=key
                    )
                except KeyError:
                    continue

            for i, line in enumerate(ax.lines, 1):
                line.set_color(plt.cm.gist_ncar(i / len(ax.lines)))
            # make legend appear on the right side of the plot, outside
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
            # plt.legend(fontsize=6)
            plt.xlim(self.min_ecc, self.max_ecc)
            plt.savefig(self.output_path / f"compare_subjects_{out_name}.png")
            plt.close()