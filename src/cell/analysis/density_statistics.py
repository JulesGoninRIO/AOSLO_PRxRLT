from collections import defaultdict
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(r'U:\Repos\aoslo_pipeline')

from src.cell.analysis.baseline_characteristics import BaselineCharacteristics
from src.cell.analysis.constants import RESULT_NAME
from src.pdf_report.global_density_pdf_report import GlobalDensityPDFReport
from src.cell.analysis.helpers import gather_results
from src.configs.parser import Parser
from src.plotter.density_statistics_plotter import DensityStatisticsPlotter
from src.shared.helpers.direction import Direction
class DensityStatistics:
    """
    A class to perform density statistics analysis.

    Attributes
    ----------
    base_path : Path
        The parent directory path.
    output_path : Path
        The output directory path.
    step : int
        The step value for processing.
    dir_to_process : Path, optional
        The specific directory to process (default is None).
    result_list : list
        A list to store results.
    resulting_df : pd.DataFrame
        A DataFrame to store the resulting data.
    density_statistics_plotter : DensityStatisticsPlotter
        An instance of DensityStatisticsPlotter for plotting results.

    Methods
    -------
    _find_subject_dirs():
        Finds and returns subject directories.
    _process_session_dir(session_path):
        Processes a session directory and returns the result DataFrame.
    gather_results():
        Gathers results from all session directories and returns the final DataFrame.
    plot():
        Plots the layer to density results.
    process_spearman_correlations():
        Processes Spearman's rank correlations.
    """

    def __init__(self, base_path, step, dir_to_process=None):
        """
        Constructs all the necessary attributes for the DensityStatistics object.

        Parameters
        ----------
        base_path : Path
            The base directory path, containing all the subjects.
        step : int
            The step value for processing.
        dir_to_process : Path, optional
            The specific directory to process (default is None).
        """
        Parser.initialize()
        self.base_path = base_path #.parent.parent
        self.output_path = self.base_path / Parser.get_global_density_dir()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.step = step
        self.dir_to_process = dir_to_process
        self.result_list = []
        self.density_statistics_plotter = DensityStatisticsPlotter(self.output_path)

    def run(self):
        df, _ = gather_results(self.base_path)
        self.plot(df)
        self.run_spearmans()

    def plot(self, resulting_df: pd.DataFrame):
        """
        Plot the layer to density results.

        This method uses the DensityStatisticsPlotter instance to plot the layer to density results
        stored in the resulting DataFrame.

        :return: None
        """
        self.density_statistics_plotter.gif_layers_to_density(self.base_path)
        self.resulting_df = resulting_df
        self.density_statistics_plotter.plot(self.resulting_df)
        self.density_statistics_plotter.plot_layer_to_density(self.resulting_df)

    def process_spearman_correlations(self):
        """
        Process Spearman's rank correlations.

        This method initializes dictionaries for Spearman's rank correlations, prepares the dataset,
        processes each directory, and reads the Spearman's rank correlation results from text files.

        :return: None
        """
        self.names = []
        spearman_dics_X = []
        spearman_dics_Y = []
        for density_a_path in self.base_path.glob(f"Subject*/Session*/{Parser.get_density_analysis_dir()}"):
            if not (density_a_path / "spearmans_X.txt").exists() or not (density_a_path / "spearmans_Y.txt").exists():
                continue
            subject = density_a_path.parent.parent.name
            session = density_a_path.parent.name
            self.names.append(f"{subject}_{session}")
            spearman_dics_X.append(json.load((density_a_path / "spearmans_X.txt").open()))
            spearman_dics_Y.append(json.load((density_a_path / "spearmans_Y.txt").open()))
        self.spearmans_dic_X = self._merge_dicts(spearman_dics_X)
        self.spearmans_dic_Y = self._merge_dicts(spearman_dics_Y)

    def _merge_dicts(self, dicts: List[Dict]):
        merged = defaultdict(lambda: defaultdict(list))
        for d in dicts:
            for key, subdict in d.items():
                for subkey, value in subdict.items():
                    merged[key][subkey].append(value)
        # Convert defaultdicts to regular dicts
        return {k: dict(v) for k, v in merged.items()}

    def save_spearman_results_to_excel(self):
        """
        Save Spearman's rank correlation results to an Excel file.

        This method saves the Spearman's rank correlation results for each category and side
        to an Excel file with separate sheets for each category.

        :return: None
        """
        writer = pd.ExcelWriter(os.path.join(self.output_path, "spearman.xlsx"), engine='xlsxwriter')
        for key in self.spearmans_dic_X.keys():
            out_csv_df = pd.DataFrame()
            out_csv_df['subjects'] = self.names
            for second_key, value in self.spearmans_dic_X[key].items():
                out_csv_df[second_key + "_X"] = value
            for second_key, value in self.spearmans_dic_Y[key].items():
                out_csv_df[second_key + "_Y"] = value
            out_csv_df.to_excel(writer, sheet_name=key)
            writer.sheets[key].set_column(1, 9, 50)
        writer.close()

    def run_spearmans(self):
        """
        Run the Spearman's rank correlation analysis.

        This method processes the Spearman's rank correlations, saves the results to an Excel file,
        plots the results, and generates various reports.

        :return: None
        """
        self.process_spearman_correlations()
        self.save_spearman_results_to_excel()
        # TODO: compute p-values
        print("DEBUG: processing spearman results")
        results_X, results_Y = self.process_spearman_results()
        print("DEBUG: plotting spearman results", results_X, results_Y)
        self.density_statistics_plotter.plot_spearmans(results_X, results_Y)
        BaselineCharacteristics(self.output_path, self.step).compare_to_baseline_characteristics()
        # too_small_areas_analyzer()
        # GlobalDensityPDFReport(self.dirs_to_process).generate_report(self.output_path)
        print("DEBUG: finished processing spearman results")

    @staticmethod
    def calculate_significant_p_values(spearmans_dic, category, side):
        """
        Calculate significant p-values for Spearman's rank correlations.

        This method calculates the significant p-values for the given category and side
        from the Spearman's rank correlation dictionary.

        :param spearmans_dic: The dictionary containing Spearman's rank correlation results.
        :type spearmans_dic: dict
        :param category: The category to analyze.
        :type category: str
        :param side: The side to analyze (e.g., 'left' or 'right').
        :type side: str
        :return: A tuple containing the list of significant values and the ratio of significant values.
        :rtype: Tuple[List[float], float]
        """
        significant_values = [value[0] for value in spearmans_dic[category][side] if len(value) > 1 and value[1] < 0.05]
        total_values = len(significant_values) + len([value[0] for value in spearmans_dic[category][side] if len(value) > 1 and value[1] > 0.05])
        ratio = len(significant_values) / total_values if total_values > 0 else 0
        return significant_values, ratio

    def process_spearman_results(self):
        """
        Process Spearman's rank correlation results.

        This method processes the Spearman's rank correlation results for various categories and sides,
        and calculates the significant p-values.

        :return: A tuple containing dictionaries of significant p-values for X and Y directions.
        :rtype: Tuple[dict, dict]
        """
        categories = ['RNFL', 'PhotoR+RPE', 'CVI', 'Choroid', 'ONL', 'INL+OPL', 'GCL+IPL', 'OS']
        sides = ['left', 'right']
        parts = ['first_part', 'second_part', 'third_part', 'fourth_part']

        results_X = {}
        results_Y = {}

        for category in categories:
            for side in sides:
                if category in self.spearmans_dic_X and side in self.spearmans_dic_X[category]:
                    results_X[f'{category}_{side}'] = DensityStatistics.calculate_significant_p_values(self.spearmans_dic_X, category, side)
                if category in self.spearmans_dic_Y and side in self.spearmans_dic_Y[category]:
                    results_Y[f'{category}_{side}'] = DensityStatistics.calculate_significant_p_values(self.spearmans_dic_Y, category, side)
            for part in parts:
                if category in self.spearmans_dic_X and part in self.spearmans_dic_X[category]:
                    results_X[f'{category}_{part}'] = DensityStatistics.calculate_significant_p_values(self.spearmans_dic_X, category, part)
                if category in self.spearmans_dic_Y and part in self.spearmans_dic_Y[category]:
                    results_Y[f'{category}_{part}'] = DensityStatistics.calculate_significant_p_values(self.spearmans_dic_Y, category, part)

        return results_X, results_Y

    def get_median_density_compare_to_median_choroid(self):
        """
        Get median density and compare to median choroid.

        This method builds a DataFrame with median, mean, and standard deviation of densities and choroid
        for each subject and session, and plots the results.

        :return: None
        """
        results = {'name': [], 'densities_X_median': [], 'densities_X_mean': [],
                'densities_X_std': [], 'Choroid_X_median': [], 'Choroid_X_mean': [],
                'Choroid_X_std': [], 'densities_Y_median': [], 'densities_Y_mean': [],
                'densities_Y_std': [], 'Choroid_Y_median': [], 'Choroid_Y_mean': [],
                'Choroid_Y_std': []}
        
        for density_a_path in self.base_path.glob(f"Subject*/Session*/{Parser.get_density_analysis_dir()}"):
            if not (density_a_path / RESULT_NAME).exists():
                continue
            subject = density_a_path.parent.parent.name
            session = density_a_path.parent.name
            try:
                result = pd.read_csv(density_a_path / RESULT_NAME, skiprows=1)
            except pd.errors.EmptyDataError:
                continue
            if result.empty:
                continue

            # drop locations after + or - 4 degrees in eccentricities
            center_result = result.drop(result[abs(result.location) > 4].index)
            center_result = center_result.drop(center_result[center_result.location == -0.0].index)
            mean_df = center_result.mean()
            median_df = center_result.median()
            sd_df = center_result.std()
            if not "Choroid_X" in median_df.keys() or not "Choroid_Y" in median_df.keys():
                continue
            results['name'].append(subject+"_"+session)
            results['densities_X_median'].append(median_df.densities_X)
            results['densities_X_mean'].append(mean_df.densities_X)
            results['densities_X_std'].append(sd_df.densities_X)
            results['densities_Y_median'].append(median_df.densities_Y)
            results['densities_Y_mean'].append(mean_df.densities_Y)
            results['densities_Y_std'].append(sd_df.densities_Y)
            results['Choroid_X_median'].append(median_df.Choroid_X)
            results['Choroid_X_mean'].append(mean_df.Choroid_X)
            results['Choroid_X_std'].append(sd_df.Choroid_X)
            results['Choroid_Y_median'].append(median_df.Choroid_Y)
            results['Choroid_Y_mean'].append(mean_df.Choroid_Y)
            results['Choroid_Y_std'].append(sd_df.Choroid_Y)

        # plot the results and save the DataFrame in csv
        results_df = pd.DataFrame(results)
        self.density_statistics_plotter.plot_median_density_compare_to_median_choroid(results_df)

        # save a csv and excel file
        writer = pd.ExcelWriter(self.out_path / 'choroid_correlation_density.xlsx', engine='xlsxwriter')
        results_df.to_excel(writer, sheet_name='results')
        writer.sheets['results'].set_column(1, 15, 20)
        writer.close()

if __name__ == "__main__":
    base_path = Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results')
    Parser.initialize()
    density_statistics = DensityStatistics(base_path, 0.1)
    density_statistics.run()