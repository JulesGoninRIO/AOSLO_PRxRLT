from typing import Dict, List, Tuple, Callable
import math
import logging
from pathlib import Path
from deprecated import deprecated
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit

from src.shared.helpers.direction import Direction
from src.configs.parser import Parser
from src.cell.analysis.constants import MM_PER_DEGREE
from src.plotter.plotter import Plotter
from src.cell.analysis.density import Density
class DensityPlotter(Plotter):
    def __init__(self, output_path: Path, step: float = 0.1):
        """
        Initialize the DensityPlotter.

        :param output_path: The path where the output will be saved.
        :type output_path: Path
        """
        self.curve_fit_limit = Parser.get_density_curve_fit_limit()
        self.step = step
        self.round_step = round(self.step * 10)
        super().__init__(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.center_limit = Parser.get_density_center_limit()
        self.lower_bound_fn = lambda c: lambda x: 5000 + 45000 * np.exp(-1 * np.abs(x - c))
        self.upper_bound_fn = lambda c: lambda x: 15_000 + 150_000 * np.exp(-0.5 * np.abs(x - c))

    def plot(
            self,
            densities: Density,
            densities_mdrnn: Density,
            densities_atms: Density
        ) -> None:
        """
        Refactored plot method to reduce duplication and improve readability.
        Plots the densities graph including raw graph, smoothed density, fitted curve,
        and the difference between MDRNN and ATMS algorithms for both x and y axes.

        :param densities: The density data for both x and y directions.
        :type densities: Density
        :param densities_mdrnn: The MDRNN density data for both x and y directions.
        :type densities_mdrnn: Density
        :param densities_atms: The ATMS density data for both x and y directions.
        :type densities_atms: Density
        """
        for axis in ['x', 'y']:
            try:
                self.plot_axis(densities, densities_mdrnn, densities_atms, axis)
            except ValueError as e:
                logging.info(f"Not enough densities in the {axis} direction to plot: {e}")

    @deprecated
    def _plot_bilateral_fitting_yellott(
            self,
            densities_pos: np.ndarray, 
            densities_neg: np.ndarray, 
            popt_pos: np.ndarray, 
            popt_neg: np.ndarray, 
            residuals_pos: np.ndarray, 
            residuals_neg: np.ndarray,
            model_pos: Callable[[np.ndarray, np.ndarray], np.ndarray],
            model_neg: Callable[[np.ndarray, np.ndarray], np.ndarray],
            direction: Direction,
            ignored_densities: np.ndarray | None = None,
            add_peak: bool | Callable | None = None,
            right_eye: bool = True
        ) -> plt.Figure:
    
        if popt_pos[3] != popt_neg[3]:
            raise ValueError('The centers of the positive and negative fits must be equal')
        center = popt_pos[3]

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[3, 1, 1])

        fig.suptitle(f"Cone Density - Healthy, {'OD' if right_eye else 'OS'}, {direction.value}-axis, Yellott's method", fontsize=22)


        exp_to_str = lambda p: f'{p[0] if p[0] > 100 else 0:.5g} + {p[1]:.5g} *\n   exp(-{p[2]:.3g} * |ecc {'−' if p[3] > 0 else '+'} {np.abs(p[3]):.3g}|)'

        if add_peak and not callable(add_peak):
            peak_height = 165_000 if not isinstance(add_peak, (int, float)) else add_peak
            add_peak = lambda modl, centr: lambda r: modl(r) + (peak_height - modl(centr)) * np.exp(-4 * np.abs(r - centr))
        if callable(add_peak):
            model_pos_peak = add_peak(model_pos, center)
            model_neg_peak = add_peak(model_neg, center)
            peak_height = model_pos_peak(center)
        
        pos_name, neg_name = ('Nasal', 'Temporal') if direction.is_X else ('Inferior', 'Superior')

        _1mm = 1 / MM_PER_DEGREE # 1mm in degrees
        # First row: density plot
        ax_density = fig.add_subplot(gs[0, :])

        neg_lim = np.round(densities_neg[:, 0].min() - self.step, self.round_step)
        pos_lim = np.round(densities_pos[:, 0].max() + self.step, self.round_step)
        x_pos = np.linspace(center, pos_lim, int((pos_lim - center)/self.step + 1))
        x_neg = np.linspace(neg_lim, center, int((center - neg_lim)/self.step + 1))

        ax_density.fill_between(np.r_[x_neg,x_pos], self.lower_bound_fn(center)(np.r_[x_neg,x_pos]), color='grey', alpha=0.1)
        ax_density.fill_between(np.r_[x_neg,x_pos], self.upper_bound_fn(center)(np.r_[x_neg,x_pos]), 300_000, color='grey', alpha=0.1)

        if ignored_densities is not None:
            ax_density.scatter(ignored_densities[:, 0], ignored_densities[:, 1], 20, 'k', alpha=0.3)
        ax_density.scatter(densities_pos[:, 0], densities_pos[:, 1], 20, 'r', alpha=0.5, label=f'{pos_name} densities')
        ax_density.scatter(densities_neg[:, 0], densities_neg[:, 1], 20, 'b', alpha=0.5, label=f'{neg_name} densities')
        if add_peak:
            x_p = x_pos[x_pos - center <= _1mm]
            x_n = x_neg[center - x_neg <= _1mm]
            ax_density.plot(x_p, model_pos(x_p), '--', color='green', alpha=1)
            ax_density.plot(x_n, model_neg(x_n), '--', color='orange', alpha=1)
            ax_density.plot(x_p, model_pos_peak(x_p), color='green', alpha=0.7)
            ax_density.plot(x_n, model_neg_peak(x_n), color='orange', alpha=0.7)
            ax_density
        x_p = x_pos[x_pos - center >= _1mm - 2*self.step] if add_peak else x_pos
        x_n = x_neg[center - x_neg >= _1mm - 2*self.step] if add_peak else x_neg
        ax_density.plot(x_p, model_pos(x_p), color='green', alpha=1, label=f'{pos_name} fit: {exp_to_str(popt_pos)}{f'\n   (fov. peak: {peak_height/1000:.0f}k)' if add_peak else ''}')
        ax_density.plot(x_n, model_neg(x_n), color='orange', alpha=1, label=f'{neg_name} fit: {exp_to_str(popt_neg)}{f'\n   (fov. peak: {peak_height/1000:.0f}k)' if add_peak else ''}')

        ax_density.legend(loc='best', fontsize=11)

        ax_density.set_xticks(np.arange(np.floor(neg_lim), np.ceil(pos_lim)+1))
        # if add_peak:
        #     ax_density.set_ylim(bottom=0, top=300_000)
        # else:
        #     ax_density.set_ylim(bottom=0, )
        ax_density.set_xlabel('Eccentricity [°]')
        ax_density.set_ylabel('Density [cells/mm²]')
        ax_density.set_yscale('log')
        yticks = ax_density.get_yticks()
        # ticks_to_add_minor = {12_500, 15_000, 17_500, 25_000}
        # ticks_to_add = {15_000, 25_000}
        ticks_to_label = {5000, 10_000, 15_000, 20_000, 25_000, 30_000, 50_000, 100_000, 200_000, 300_000}
        combined_yticks = sorted(set(yticks) | ticks_to_label)
        ax_density.set_yticks([7500, 12_500, 17_500, 40_000, 75_000, 150_000], minor=True)
        ax_density.set_yticks(combined_yticks)
        ax_density.set_yticklabels([f'{tick/1000:.0f}k' if tick in ticks_to_label else '' for tick in combined_yticks])
        ax_density.tick_params(which='minor', labelleft=False)
        ax_density.set_ylim(bottom=5000, top=300_000)
        ax_density.set_xlim(left=neg_lim, right=pos_lim)

        # Second row: Residual plots
        ax_left = fig.add_subplot(gs[1, 0])
        ax_middle = fig.add_subplot(gs[1, 1], sharey=ax_left)
        ax_right = fig.add_subplot(gs[1, 2], sharey=ax_left)

        # Residuals scatter plot
        ax_left.scatter(densities_pos[:, 0], residuals_pos, 20, 'r', alpha=0.5)
        ax_left.scatter(-densities_neg[::-1, 0], residuals_neg[::-1], 20, 'b', alpha=0.5)
        ax_left.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_left.set_xticks(np.arange(0, 11))
        ax_left.legend([pos_name, neg_name])
        ax_left.set_xlabel('|Eccentricity| [°]')
        ax_left.set_ylabel('Residuals [cells/mm²]')

        # Residuals histogram
        ax_middle.hist([residuals_pos, residuals_neg], 
                        color=['r', 'b'], 
                        orientation='horizontal', 
                        bins=20, 
                        alpha=0.5)
        ax_middle.set_xlabel('Frequency')

        # Residuals boxplot
        boxes = ax_right.boxplot([residuals_pos, residuals_neg], 
                                    vert=True, 
                                    patch_artist=True, 
                                    medianprops=dict(color="black"), 
                                    widths=0.4)
        for box, color in zip(boxes['boxes'], ['r', 'b']):
            box.set_facecolor(color)
            box.set_alpha(0.5)
        ax_right.set_xticklabels([pos_name[:3], neg_name[:3]])

        plt.tight_layout()
        plt.savefig(self.output_path / f"cone_density_curve_fitted_{direction.value}.png")
        plt.close()

    def plot_bilateral_fitting_yellott(
            self,
            densities_neg: np.ndarray, 
            densities_pos: np.ndarray, 
            center: float,
            popt_neg: np.ndarray, 
            popt_pos: np.ndarray, 
            lower_bound: Callable[[np.ndarray], np.ndarray],
            upper_bound: Callable[[np.ndarray], np.ndarray],
            model: Callable[[np.ndarray, np.ndarray], np.ndarray],
            direction: Direction,
            ignored_densities: np.ndarray | None = None,
            right_eye: bool = True
        ) -> plt.Figure:

        fig = plt.figure(figsize=(10, 6))

        fig.suptitle(f"Cone Density - Healthy, {'OD' if right_eye else 'OS'}, {direction.value}-axis, Yellott's method", fontsize=22)

        exp_to_str = lambda p: \
            fr'$\exp[ {p[0]:.3g} - {p[1]:.3g} \cdot |ecc {'−' if center > 0 else '+'} {np.abs(center):.3g}|$' + '\n' \
          + fr'      $+ {p[3]:.3g} / (|ecc {'−' if center > 0 else '+'} {np.abs(center):.3g}| + {p[2]:.3g}) ]$'

        neg_name, pos_name= ('Temporal', 'Nasal') if direction.is_X else ('Superior', 'Inferior')

        ax_density = fig.add_subplot(1,1,1)

        neg_lim = -10
        pos_lim = 10
        x_neg = np.linspace(neg_lim, center, 500)
        x_pos = np.linspace(center, pos_lim, 500)

        ax_density.fill_between(np.r_[x_neg,x_pos], lower_bound(np.r_[x_neg,x_pos]), color='grey', alpha=0.1)
        ax_density.fill_between(np.r_[x_neg,x_pos], upper_bound(np.r_[x_neg,x_pos]), 330_000, color='grey', alpha=0.1)

        if ignored_densities is not None:
            ax_density.scatter(ignored_densities[:, 0], ignored_densities[:, 1], 20, 'k', alpha=0.3)
        ax_density.scatter(densities_neg[:, 0], densities_neg[:, 1], 20, 'b', alpha=0.5, label=f'{neg_name} densities')
        ax_density.scatter(densities_pos[:, 0], densities_pos[:, 1], 20, 'r', alpha=0.5, label=f'{pos_name} densities')
        ax_density.plot(x_neg, model(x_neg), color='orange', alpha=1, label=f'{neg_name} fit:\n{exp_to_str(popt_neg)}')
        ax_density.plot(x_pos, model(x_pos), color='green', alpha=1, label=f'{pos_name} fit:\n{exp_to_str(popt_pos)}')
        # add text at location of the peak displaying the peak value
        ax_density.text(center, model(center), f'{model(center)/1000:.0f}k  ', fontsize=11, color='black', ha='right', va='center')


        ax_density.legend(loc='best', fontsize=11)

        ax_density.set_xticks(np.arange(np.floor(neg_lim), np.ceil(pos_lim)+1))
        ax_density.set_xlabel('Eccentricity [°]')
        ax_density.set_ylabel('Density [cells/mm²]')
        ax_density.set_yscale('log')
        yticks = ax_density.get_yticks()
        # ticks_to_add_minor = {12_500, 15_000, 17_500, 25_000}
        # ticks_to_add = {15_000, 25_000}
        ticks_to_label = {5000, 10_000, 15_000, 20_000, 25_000, 30_000, 50_000, 100_000, 200_000, 300_000}
        combined_yticks = sorted(set(yticks) | ticks_to_label)
        ax_density.set_yticks([7500, 12_500, 17_500, 40_000, 75_000, 150_000], minor=True)
        ax_density.set_yticks(combined_yticks)
        ax_density.set_yticklabels([f'{tick/1000:.0f}k' if tick in ticks_to_label else '' for tick in combined_yticks])
        ax_density.tick_params(which='minor', labelleft=False)
        ax_density.set_xlim(left=neg_lim, right=pos_lim)
        ax_density.set_ylim(bottom=5000, top=330_000)

        plt.tight_layout()
        plt.savefig(self.output_path / f"cone_density_curve_fitted_new_{direction.value}.png")
        plt.close()

    def plot_density_curve_smoothed(self, densities_smthd_x: np.ndarray, densities_smthd_y: np.ndarray):
        plt.plot(densities_smthd_x[:,0], densities_smthd_x[:,1], color='r', linewidth=1, label=f'Smoothed densities X-axis')
        plt.plot(densities_smthd_y[:,0], densities_smthd_y[:,1], color='b', linewidth=1, label=f'Smoothed densities Y-axis')
        plt.legend()
        plt.xlim(-10, 10)
        plt.xticks(np.arange(-10, 11))
        plt.savefig(self.output_path / f"cone_density_curve_smoothed.png")
        plt.close()

    def plot_density_curve_fitting(
            self,
            densities_avg_x: np.ndarray, 
            densities_avg_y: np.ndarray, 
            popt_x: np.ndarray, 
            popt_y: np.ndarray, 
            residuals_x: np.ndarray, 
            residuals_y: np.ndarray,
            exp_model: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[3, 1, 1])

        fig.suptitle('Cone Density Curve Fitting', fontsize=24)

        exp_to_str = lambda p: f'{p[0] if p[0] > 500 else 0:.5g} + {p[1]:.5g} * exp(-{p[2]:.4g} * |ecc|)'

        exp_model_fov_peak = lambda x, p: exp_model(x, p) + ((160000 - p[0] - p[1]) * np.exp(-5 * x) if p[0] + p[1] < 160000 else 0)

        # First row: density plot
        ax_density = fig.add_subplot(gs[0, :])
        ax_density.scatter(densities_avg_x[:, 0], densities_avg_x[:, 1], 20, 'r', alpha=0.5)
        ax_density.scatter(densities_avg_y[:, 0], densities_avg_y[:, 1], 20, 'b', alpha=0.5)
        x_data = np.linspace(0, 10, int(10/self.step + 1))
        ax_density.plot(x_data, exp_model_fov_peak(x_data, popt_x), 'r', alpha=1)
        ax_density.plot(x_data, exp_model_fov_peak(x_data, popt_y), 'b', alpha=1)
        ax_density.legend(['X-axis sampled densities', 'Y-axis sampled densities', f'X-axis fit: {exp_to_str(popt_x)} (+ fov. peak)', f'Y-axis fit: {exp_to_str(popt_y)} (+ fov. peak)'])
        ax_density.set_xticks(np.arange(0, 11))
        ax_density.set_ylim(bottom=0, top=160000)
        ax_density.set_xlabel('|Eccentricity| [°]')
        ax_density.set_ylabel('Density [cells/mm²]')

        # Second row: Residual plots
        ax_left = fig.add_subplot(gs[1, 0])
        ax_middle = fig.add_subplot(gs[1, 1], sharey=ax_left)
        ax_right = fig.add_subplot(gs[1, 2], sharey=ax_left)

        # Residuals scatter plot
        ax_left.scatter(densities_avg_x[:, 0], residuals_x, 20, 'r', alpha=0.5)
        ax_left.scatter(densities_avg_y[:, 0], residuals_y, 20, 'b', alpha=0.5)
        ax_left.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_left.set_xticks(np.arange(1, 11))
        ax_left.legend(['X-axis', 'Y-axis'])
        ax_left.set_xlabel('|Eccentricity| [°]')
        ax_left.set_ylabel('Residuals [cells/mm²]')

        # Residuals histogram
        ax_middle.hist([residuals_x, residuals_y], 
                       color=['r', 'b'], 
                       orientation='horizontal', 
                       bins=20, 
                       alpha=0.5)
        ax_middle.set_xlabel('Frequency')

        # Residuals boxplot
        boxes = ax_right.boxplot([residuals_x, residuals_y], 
                                 vert=True, 
                                 patch_artist=True, 
                                 medianprops=dict(color="black"), 
                                 widths=0.4)
        for box, color in zip(boxes['boxes'], ['r', 'b']):
            box.set_facecolor(color)
            box.set_alpha(0.5)
        ax_right.set_xticklabels(['X-axis', 'Y-axis'])

        plt.tight_layout()
        plt.savefig(self.output_path / f"cone_density_curve_fitting.png")
        plt.close()

    def plot_sampled_densities_axis(self, densities_mdrnn: np.ndarray, densities_atms: np.ndarray, densities_avg: np.ndarray, density_smthd: np.ndarray, direction: Direction) -> np.ndarray:

        ticks = np.arange(math.floor(min(density_smthd[:, 0])), math.ceil(max(density_smthd[:, 0])) + 1, 1)

        plt.plot(densities_atms[:,0], densities_atms[:,1], 'ro-', label='ATMS')
        plt.plot(densities_mdrnn[:, 0], densities_mdrnn[:, 1], 'bo-', label='MDRNN')
        plt.plot(densities_avg[:, 0], densities_avg[:, 1], 'k--', label='Weighted avg')
        plt.plot(density_smthd[:, 0], density_smthd[:, 1], '-', color='lime', label='Smoothed')
        plt.legend(fontsize=20)
        plt.xticks(ticks, ticks, fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel("Retinal Eccentricity [°]", fontsize=20)
        plt.ylabel("Cone Density [cones/mm²]", fontsize=20)
        plt.title(f"Density curves for {direction.value} axis", fontsize=24)
        plt.savefig(self.output_path / f"sampling_density_curves_{direction.value}.png")
        plt.close()

    def plot_axis(
        self,
        densities: Density,
        densities_mdrnn: Density,
        densities_atms: Density,
        axis: str) -> None:
        """
        Helper method to plot for a single axis (x or y).

        :param densities: The density data for both x and y directions.
        :param densities_mdrnn: The MDRNN density data for both x and y directions.
        :param densities_atms: The ATMS density data for both x and y directions.
        :param axis: The axis to plot ('x' or 'y').
        """
        axis_data = getattr(densities, axis)
        gaps = self.get_gaps(axis_data)
        lowess = sm.nonparametric.lowess(
            list(axis_data.values()),
            list(axis_data.keys()),
            frac=self.step
        )
        ticks = np.arange(math.floor(min(lowess[:, 0])), math.ceil(max(lowess[:, 0])) + 1, 1)
        self.plot_raw(axis_data, ticks, axis)
        first_part, second_part = self.plot_smoothed(lowess, gaps, axis_data, ticks, axis)
        self.plot_fitted_curve(first_part, second_part, ticks, axis)
        self.plot_differences_algorithms(
            lowess,
            getattr(densities_mdrnn, axis),
            getattr(densities_atms, axis), ticks, axis)

    def plot_raw(self, densities: Dict[str, float], tick: np.array, axis: str) -> None:
        """
        Plot the raw density for an axis

        :param densities: the dictionnary with the raw density
        :type densities: Dict[str, float]
        :param tick: the array with the locations of the density
        :type tick: np.array
        :param axis: the axis we have data from
        :type axis: str
        """

        plt.plot(
            np.array(list(densities.keys())),
            np.array(list(densities.values())))
        plt.xticks(tick, tick, fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel("Retinal Eccentricity [°]", fontsize=20)
        plt.ylabel("Cone Density [cones/mm²]", fontsize=20)
        plt.title(f"Raw density for {axis}-axis", fontsize=24)
        plt.savefig(self.output_path / f"density_{axis}_raw.png")
        plt.close()

    def plot_smoothed(
        self,
        lowess: np.ndarray,
        gaps: List[List[int]],
        densities: Dict[str, float],
        tick: np.array, axis: str) -> Tuple[List[np.array], List[np.array]]:
        """
        Plot the smoothed curve of the raw density

        :param lowess: the smoothed density
        :type lowess: np.ndarray
        :param gaps: the gaps (>2) in the densities to hide from plot
        :type gaps: List[List[int]]
        :param densities: the density dictionnary
        :type densities: Dict[str, float]
        :param tick: the locations of the density values
        :type tick: np.array
        :param axis: the axis we have data from
        :type axis: str
        :return: the left and right part of the smoothed curve
        :rtype: Tuple[List[np.array], List[np.array]]
        """

        # set up the limits of the indices around the center (higher += 1)
        min_indice = 0
        try:
            lower_indice = np.max(np.argwhere(
                lowess[:, 0] <= -self.center_limit+self.step))
        except ValueError:
            lower_indice = np.argmin(lowess[:, 0])
        try:
            higher_indice = np.min(np.argwhere(
                lowess[:, 0] >= self.center_limit))
        except ValueError:
            higher_indice = np.argmax(lowess[:, 0])

        # separate the curve in the 2 side with center removed
        first_part = [lowess[:, 0][0:lower_indice], lowess[:, 1][0:lower_indice]]
        second_part = [lowess[:, 0][higher_indice:], lowess[:, 1][higher_indice:]]

        # separate the curves in multiple ones if there are gaps
        list_lines = []
        if len(gaps) > 1:
            first_done = False
            second_done = False
            for i in range(len(gaps)-1):
                lower_limit = np.argwhere(np.array(list(densities.keys())) == np.round(
                    gaps[i][0]-self.step*gaps[i][1], decimals=1))[0][0]
                higher_limit = np.argwhere(np.array(list(densities.keys())) == np.round(
                    gaps[i+1][0]-self.step*gaps[i+1][1], decimals=1))[0][0]
                if higher_limit < 0:
                    first_done = True
                    list_lines.append([
                        first_part[0][lower_limit:higher_limit],
                        first_part[1][lower_limit:higher_limit]])
                else:
                    second_done = True
                    list_lines.append([
                        second_part[0][lower_limit:higher_limit],
                        second_part[1][lower_limit:higher_limit]])
            if not first_done:
                list_lines.append(first_part)
            if not second_done:
                list_lines.append(second_part)
        else:
            list_lines.append(first_part)
            list_lines.append(second_part)

        # and plot the lines
        for line in list_lines:
            plt.plot(line[0], line[1], color="orange")
        plt.xticks(tick, tick, fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel("Retinal Eccentricity [°]", fontsize=20)
        plt.ylabel("Cone Density [cones/mm²]", fontsize=20)
        plt.title(f"Smoothed density for {axis}-axis", fontsize=24)
        plt.savefig(self.output_path / f"density_{axis}_smoothed.png")
        plt.close()

        return first_part, second_part

    def plot_fitted_curve(
        self,
        first_part: List[np.array],
        second_part: List[np.array],
        tick: np.array,
        axis: str) -> None:
        """
        Plot the smoothed density with a fitted curve in the center so that
        we approximate the value of the cone density in the center because we know
        the ATMS algorithm cannot detect all the cones in the very center of the eye.
        This function is set up to find a logarithm curve in the center but we can
        also search for a linear one by uncommenting the code.

        :param first_part: the left part of the smoothed curve
        :type first_part: List[np.array]
        :param second_part: the right part of the smoothed curve
        :type second_part: List[np.array]
        :param tick: the location values of the densities
        :type tick: np.array
        :param axis: the axis we have data from (either x or y)
        :type axis: str
        """

        def func_pos_log(a: int, b: int, c: int, x: float) -> float:
            """
            Positive logarithm function

            :param a: parameter of the log function
            :type a: int
            :param b: parameter of the log function
            :type b: int
            :param c: parameter of the log function
            :type c: int
            :param x: value of the log function
            :type x: float
            :return: the result of the log function
            :rtype: float
            """
            return a-b*np.log(c*x)

        def func_pos_lin(a: int, b: int, x: float) -> float:
            """
            Positive linear function

            :param a: parameter of the linear function
            :type a: int
            :param b: parameter of the linear function
            :type b: int
            :param x: value of the linear function
            :type x: float
            :return: the result of the linear function
            :rtype: float
            """
            return a*x + b

        def func_neg_log(a: int, b: int, c: int, x: float) -> float:
            """
            Negative logarithm function

            :param a: parameter of the log function
            :type a: int
            :param b: parameter of the log function
            :type b: int
            :param c: parameter of the log function
            :type c: int
            :param x: value of the log function
            :type x: float
            :return: the result of the log function
            :rtype: float
            """
            return a-b*np.log(-c*x)

        def func_neg_lin(a: int, b: int, x: float) -> float:
            """
            Positive linear function

            :param a: parameter of the linear function
            :type a: int
            :param b: parameter of the linear function
            :type b: int
            :param x: value of the linear function
            :type x: float
            :return: the result of the linear function
            :rtype: float
            """
            return -a*x + b

        try:
            fit_lower_indice = np.min(np.argwhere(
                first_part[0] <= -self.curve_fit_limit))
        except ValueError:
            fit_lower_indice = 0
        try:
            fit_higher_indice = np.max(np.argwhere(
                second_part[0] >= self.curve_fit_limit+self.step))
        except ValueError:
            fit_higher_indice = 4
        try:
            # fit the left curve with the log function
            first_part_fit_curve_log = curve_fit(
                    lambda x, a, b, c: func_neg_log(a, b, c, x),
                    first_part[0][fit_lower_indice:],
                    first_part[1][fit_lower_indice:],
                    p0=[150000, 15000, 80])
            # first_part_fit_curve_lin = curve_fit(lambda x,a,b,c: func_neg_lin(a,b,x),
            #                                      first_part[0][fit_lower_indice:],
            #                                      first_part[1][fit_lower_indice:])
            left_part = np.arange(-self.curve_fit_limit, 0, self.step)
            left_part_log = np.array([
                func_neg_log(
                    first_part_fit_curve_log[0][0],
                    first_part_fit_curve_log[0][1],
                    first_part_fit_curve_log[0][2], t) for t in left_part])
        except (ValueError, TypeError, RuntimeError):
            logging.error(
                "Cannot plot the left fitted curve because we don't have enough data.")
            first_part_fit_curve_log = 0
            left_part = [0]
            left_part_log = [0]
        try:
            # fit the right curve with the log function
            second_part_fit_curve_log = curve_fit(
                lambda x, a, b, c: func_pos_log(a, b, c, x),
                second_part[0][:fit_higher_indice],
                second_part[1][:fit_higher_indice], p0=[150000, 15000, 80])
            # second_part_fit_curve_lin = curve_fit(lambda x,a,b,c: func_pos_lin(a,b,x),
            #                                       second_part[0][:fit_higher_indice],
            #                                       second_part[1][:fit_higher_indice])
            right_part = np.arange(
                self.step, self.curve_fit_limit+self.step, self.step)
            right_part_log = np.array([func_pos_log(second_part_fit_curve_log[0][0],
                                                    second_part_fit_curve_log[0][1],
                                                    second_part_fit_curve_log[0][2],
                                                    t) for t in right_part])
        except (ValueError, TypeError, RuntimeError):
            logging.error(
                "Cannot plot the first curve because we don't have enough data.")
            second_part_fit_curve_log = 0
            right_part = [0]
            right_part_log = [0]

        # left_part_lin = np.array([func_neg_lin(first_part_fit_curve_lin[0][0],
        #                                        first_part_fit_curve_lin[0][1],
        #                                        t) for t in left_part])
        # right_part_lin = np.array([func_pos_lin(second_part_fit_curve_lin[0][0],
        #                                         second_part_fit_curve_lin[0][1],
        #                                         t) for t in right_part])

        # plot the smoothed curve and the fitted on in the center
        plt.plot(left_part, left_part_log, linestyle='dashed', color="blue")
        plt.plot(right_part, right_part_log, linestyle='dashed', color="blue")
        plt.plot(first_part[0], first_part[1], color="orange")
        plt.plot(second_part[0], second_part[1], color="orange")
        plt.xticks(tick, tick, fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel("Retinal Eccentricity [°]", fontsize=20)
        plt.ylabel("Cone Density [cones/mm²]", fontsize=20)
        plt.title(f"Fitted curve density for {axis}-axis", fontsize=24)
        plt.savefig(self.output_path / f"density_{axis}_fitted.png")
        plt.close()

    def plot_differences_algorithms(self,
                                    lowess: np.ndarray,
                                    densities_mdrnn: Dict[str, float],
                                    densities_atms: Dict[str, float],
                                    tick: np.array,
                                    axis: str) -> None:
        """
        Plot the difference between MDRNN and ATMS algorithm in the overlapping
        region (= the locations where we get densities from both algorithms)

        :param lowess: the smoothed density
        :type lowess: np.ndarray
        :param densities_mdrnn: the raw density from MDRNN algorithm
        :type densities_mdrnn: Dict[str, float]
        :param densities_atms: the raw density from ATMS algorithm
        :type densities_atms: Dict[str, float]
        :param tick: the locations of the densities
        :type tick: np.array
        :param axis: the axis we have data from (either x or y)
        :type axis: str
        """

        # Next we want 3 graphs in one: the original one, the one where davidson
        # MDRNN is taken when overlapping, the one where mikhail ATMS is taken
        # when overlapping

        # first plot the "normal" curve (=mean of both algorithms)
        plt.plot(lowess[:, 0], lowess[:, 1], color="black", label="Mean")

        # then smooth and plot ATMS density
        lowess_atms = sm.nonparametric.lowess(
            list(densities_atms.values()), list(densities_atms.keys()), frac=self.step)
        plt.plot(lowess_atms[:, 0], lowess_atms[:, 1], color="red", label="ATMS")

        # same for MDRNN density
        lowess_mdrnn = sm.nonparametric.lowess(list(densities_mdrnn.values()), list(
            densities_mdrnn.keys()), frac=self.step)
        plt.plot(lowess_mdrnn[:, 0], lowess_mdrnn[:, 1], color="blue", label="MDRNN")

        plt.legend(fontsize=20)
        plt.xticks(tick, tick, fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel("Retinal Eccentricity [°]", fontsize=20)
        plt.ylabel("Cone Density [cones/mm²]", fontsize=20)
        plt.title(
            f"Difference density curves for algorithms for {axis}-axis", fontsize=24)
        plt.savefig(self.output_path / f"density_{axis}_differences.png")
        plt.close()


    def get_gaps(self, densities: Dict[str, float]) -> List[List[int]]:
        """
        Get where we have a gap (= no values) in the densities

        :param densities: the densities of the subject
        :type densities: Dict[str, float]
        :return: the list of gaps that are bigger than 2 values
        :rtype: List[List[int]]
        """

        try:
            smaller_key = np.min(np.array(list(densities.keys())))
            bigger_key = np.max(np.array(list(densities.keys())))
        except ValueError:
            # we don't have enough values
            raise
        supposed_keys = np.round(
            np.arange(smaller_key, bigger_key, self.step), decimals=1)
        gaps = [[smaller_key, 0]]
        followed_key = 0
        for supposed_key in supposed_keys:
            if supposed_key in densities.keys():
                if followed_key > 2:
                    gaps.append([supposed_key-self.step, followed_key])
                followed_key = 0
            else:
                followed_key += 1
        gaps.append([bigger_key, 0])

        return gaps