from typing import Callable, Dict, List, Tuple
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from src.cell.analysis.constants import RESULT_NAME
from src.configs.parser import Parser
from src.plotter.baseline_characteristics_plotter import BaselineCharacteristicsPlotter

type SubjectData = Tuple[int, float, float, str, datetime, pd.DataFrame]
type SubjectsDict = Dict[str, SubjectData]
type StatisticsResult = Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]

class BaselineCharacteristics:
    """
    A class used to represent the baseline characteristics of patients.
    """

    def __init__(self, output_path: Path, step: float) -> None:
        """
        Initialize the BaselineCharacteristics class.

        :param output_path: The path where output files will be saved
        :type output_path: Path
        :param step: The step size used in calculations
        :type step: float
        :return: None
        """
        self.output_path = output_path
        self.step = step
        self.axial_length_file = Parser.get_axial_length_file()
        self.resolution = round((2 * 10) / self.step + 1)

    def compare_to_baseline_characteristics(self) -> None:
        """
        Look for patients difference between characteristics in patients (age,
        axial densities, spherical equivalence, sex, date of visit).
        """
        healthy_data = self.get_patient_data()
        subjects_dirs = self.gather_parameters(healthy_data)
        self.write_results(subjects_dirs)
        self.initialize_parameters()

        self.baseline_characteristics_plotter = BaselineCharacteristicsPlotter(
            self.output_path, self.min_ecc, self.max_ecc, self.eccentricity_bins
        )
        
        self.process_age_statistics(subjects_dirs)
        self.process_axial_statistics(subjects_dirs)
        self.process_spherical_statistics(subjects_dirs)
        self.process_sex_statistics(subjects_dirs)
        self.process_time_visits_statistics(subjects_dirs)

        self.baseline_characteristics_plotter.plot_subject_against_subject(subjects_dirs)

    def process_age_statistics(self, subjects_dirs: SubjectsDict) -> None:
        """
        Process age-related statistics and plot the results.

        :param subjects_dirs: Directories of subjects to process.
        :type subjects_dirs: list
        """
        age_median_dfs, age_std_dfs, age_range_dfs = self.get_age_statistics(subjects_dirs)

        label_callback = lambda i, n: f"{self.age_bins[i]}≤age<{self.age_bins[i+1]} (n={round(n / self.resolution)})"
        self.baseline_characteristics_plotter.plot_characteristics('age', age_median_dfs, age_range_dfs, label_callback)
        self.save_statistics('age', age_median_dfs, age_std_dfs, age_range_dfs, label_callback)
        self.baseline_characteristics_plotter.plot_errorbars(age_median_dfs, age_std_dfs, age_range_dfs, label_callback)

    def process_axial_statistics(self, subjects_dirs: SubjectsDict):
        """
        Process axial length statistics and plot the results.

        :param subjects_dirs: Directories of subjects to process.
        :type subjects_dirs: list
        """
        axial_median_dfs, axial_std_dfs, axial_range_dfs = self.get_axial_statistics(subjects_dirs)

        label_callback = lambda i, n: f"{self.axial_length_bins[i]}≤axial length<{self.axial_length_bins[i+1]} (n={round(n / self.resolution)})"
        self.baseline_characteristics_plotter.plot_characteristics('axial_length', axial_median_dfs, axial_range_dfs, label_callback)
        self.save_statistics('axial', axial_median_dfs, axial_std_dfs, axial_range_dfs, label_callback)

    def process_spherical_statistics(self, subjects_dirs: SubjectsDict) -> None:
        """
        Process spherical equivalence statistics and plot the results.

        :param subjects_dirs: Directories of subjects to process.
        :type subjects_dirs: list
        """
        spherical_median_dfs, spherical_std_dfs, spherical_range_dfs = self.get_spherical_statistics(subjects_dirs)

        label_callback = lambda i, n: f"{self.spherical_equiv_bins[i][0]}≤spherical equivalence≤{self.spherical_equiv_bins[i][1]} (n={round(n / self.resolution)})"
        self.baseline_characteristics_plotter.plot_characteristics('spherical', spherical_median_dfs, spherical_range_dfs, label_callback)
        self.save_statistics('spherical', spherical_median_dfs, spherical_std_dfs, spherical_range_dfs, label_callback)

    def process_sex_statistics(self, subjects_dirs: SubjectsDict) -> None:
        """
        Process sex-related statistics and plot the results.

        :param subjects_dirs: Directories of subjects to process.
        :type subjects_dirs: list
        """
        sex_median_dfs, sex_std_dfs, sex_range_dfs = self.get_sex_statistics(subjects_dirs)
        label_callback = lambda i, n: f"sex = {'M' if i == 0 else 'F'} (n={round(n / self.resolution)})"
        self.baseline_characteristics_plotter.plot_characteristics('sex', sex_median_dfs, sex_range_dfs, label_callback)
        self.save_statistics('sex', sex_median_dfs, sex_std_dfs, sex_range_dfs, label_callback)

    def process_time_visits_statistics(self, subjects_dirs: SubjectsDict) -> None:
        """
        Process time visits statistics and plot the results.

        :param subjects_dirs: Directories of subjects to process.
        :type subjects_dirs: list
        """
        visit_date_median_dfs, visit_date_std_dfs, visit_date_range_dfs = self.get_time_visits_statistics(subjects_dirs)

        date = lambda i: self.visit_dates_bins[i].strftime('%d/%m/%Y')
        label_callback = lambda i, n: f'{(
                 f"date of visit<{date(0)}" if i == 0 
            else f"{date(0)}≤date of visit≤{date(1)}" if i == 1 
            else f"date of visit>{date(1)}"
        )} (n={round(n / self.resolution)})'
        self.baseline_characteristics_plotter.plot_characteristics('visit_date', visit_date_median_dfs, visit_date_range_dfs, label_callback)
        self.save_statistics('visit_date', visit_date_median_dfs, visit_date_std_dfs, visit_date_range_dfs, label_callback)

    def get_patient_data(self) -> pd.DataFrame:
        """
        Retrieve patient data from the axial length file.

        :return: pd.DataFrame containing data of healthy patients.
        :rtype: pd.DataFrame
        """
        all_data = pd.ExcelFile(self.axial_length_file)
        dfs = {sheet_name: all_data.parse(sheet_name) for sheet_name in all_data.sheet_names}
        return dfs['Healthy']

    def gather_parameters(self, healthy_data: pd.DataFrame) -> SubjectsDict:
        """
        Gather parameters such as age, axial length, spherical equivalence, sex, and visit date from healthy patient data.

        :param healthy_data: pd.DataFrame containing healthy patient data.
        :type healthy_data: pd.DataFrame
        :return: Dictionary with patient IDs as keys and tuples of parameters as values.
        :rtype: dict
        """
        patient_ids_to_process = healthy_data['AOSLO ID']
        birth_dates = healthy_data['DDN']
        visit_dates = healthy_data['Date of visit']
        ages = (visit_dates - birth_dates).dt.days / 365
        ages = ages[pd.notna(ages)]
        axial_lengths = healthy_data['AL D (mm)'].where(healthy_data['Laterality'] == 'OD', healthy_data['AL G (mm)'])
        spherical_equivs = healthy_data['Equi Sph D'].where(healthy_data['Laterality'] == 'OD', healthy_data['Equi Sph G'])
        sexes = healthy_data['Sexe']

        subjects_dirs = {}
        for patient_id, age, axial_length, spherical_equiv, sex, visit_date in \
                zip(patient_ids_to_process, ages, axial_lengths, spherical_equivs, sexes, visit_dates):
            key = f'Subject{patient_id}'
            try:
                subject_df = BaselineCharacteristics.gather_raw_result(self.output_path.parent, key)
            except (IndexError, UnboundLocalError):
                continue
            subjects_dirs[key] = (age, axial_length, spherical_equiv, sex, visit_date, subject_df)
        return subjects_dirs

    @staticmethod
    def gather_raw_result(parent_path: Path, subject_name: str) -> pd.DataFrame:
        """
        Gather the results for one subject that lies in the parent directory
        and create a pd.DataFrame from it

        :param parent_path: the base path where all the subject lies
        :type parent_path: Path
        :param subject_name: the name of the subject to put into the pd.DataFrame
        :type subject_name: str
        :return: the results in a pd.DataFrame
        :rtype: pd.DataFrame
        """

        for result_file in parent_path.glob(f'{subject_name}/Session*/{Parser.get_density_analysis_dir()}/{RESULT_NAME}'):
            try:
                result = pd.read_csv(result_file, skiprows=1)
            except pd.errors.EmptyDataError:
                continue
            if result.empty:
                continue
            return result[abs(result['location']) <= 10]

    def write_results(self, subjects_dirs: SubjectsDict) -> None:
        """
        Compute median & std over all subjects for each location and write the results to an Excel file.

        :param subjects_dirs: Dictionary with patient IDs as keys and tuples of parameters as values.
        """
        result_df_range = pd.concat([infos[-1] for infos in subjects_dirs.values()])
        result_df_median, result_df_std = self._get_median_std(result_df_range)
        write_df = pd.DataFrame()
        for column in result_df_median.columns:
            if column == 'location':
                continue
            write_df[column] = self._combine_median_std(result_df_median[column], result_df_std[column])

        writer = pd.ExcelWriter(os.path.join(self.output_path, 'results_median_std.xlsx'), engine='xlsxwriter')
        write_df.to_excel(writer, sheet_name='results')
        writer.sheets['results'].set_column(1, 18, 50)
        writer.close()

    def initialize_parameters(self) -> None:
        """
        Initialize the parameters for age bins, axial length bins, spherical equivalence bins, visit dates bins,
        and eccentricity bins.
        """
        self.age_bins = np.array([20, 35, 50, 65])
        self.axial_length_bins = np.array([21.0, 23.0, 24.0, 25.0, 29.0])
        self.spherical_equiv_bins = [(-7.12, -3.12), (-1.5, -1.12), (2.12, 5.37)]
        self.visit_dates_bins = [datetime(2021, 7, 1), datetime(2022, 1, 1)]
        self.min_ecc = -10
        self.max_ecc = 10
        self.eccentricity_bins = np.linspace(self.min_ecc, self.max_ecc, self.resolution)

    def _get_median_std(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the median and standard deviation of a pd.DataFrame.

        :param df: pd.DataFrame to calculate statistics for.
        :type df: pd.DataFrame
        :return: pd.DataFrames containing median and standard deviation values.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        groupped = df.groupby(df.location)
        return groupped.median(), groupped.std()

    def _combine_median_std(self, median_s: pd.Series, std_s: pd.Series) -> pd.Series:
        """
        Combine median and standard deviation values into one pd.Series.

        :param median_s: pd.Series containing median values.
        :type median_s: pd.Series
        :param std_s: pd.Series containing standard deviation values.
        :type std_s: pd.Series
        :return: pd.Series containing median and standard deviation values.
        :rtype: pd.Series
        """
        return median_s.combine(std_s, lambda x, y: f'{x:.6g} ({y:.6g})')
    
    def save_statistics(
            self,
            characteristic: str,
            median_dfs: List[pd.DataFrame],
            std_dfs: List[pd.DataFrame],
            range_dfs: List[pd.DataFrame],
            label_callback: Callable[[int, int], str], 
            col_width: int = 50
        ) -> None:
        """
        Save the calculated statistics to an Excel file.
        """
        writer = pd.ExcelWriter(os.path.join(self.output_path, f'{characteristic}_median_std.xlsx'), engine='xlsxwriter')
        for column in median_dfs[0].columns:
            if column == 'location':
                continue
            out_csv_df = pd.DataFrame()
            for i, (median_df, std_df, range_df) in enumerate(zip(median_dfs, std_dfs, range_dfs)):
                out_csv_df[label_callback(i, len(range_df))] = self._combine_median_std(median_df[column], std_df[column])
            out_csv_df.to_excel(writer, sheet_name=column)

            # Set column width
            writer.sheets[column].set_column(1, len(median_dfs), col_width)
        writer.close()

    def get_age_statistics(
            self,
            subjects_dirs: SubjectsDict
        ) -> StatisticsResult:
        """
        Calculate median and standard deviation statistics for different age ranges.
        
        :param subjects_dirs: Dictionary with patient IDs as keys and tuples of parameters as values.
        :type subjects_dirs: Dict[str, Tuple[int, float, float, str, datetime, pd.DataFrame]]
        :return: Tuple containing pd.DataFrames for medians and standard deviations of different age ranges.
        :rtype: Tuple[pd.DataFrame]
        """
        first_age_range_df = pd.DataFrame()
        second_age_range_df = pd.DataFrame()
        third_age_range_df = pd.DataFrame()
        for subject_id in subjects_dirs:
            age, _, _, _, _, subject_df = subjects_dirs[subject_id]
            if self.age_bins[0] <= age < self.age_bins[1]:
                first_age_range_df = pd.concat([first_age_range_df, subject_df])
            elif self.age_bins[1] <= age < self.age_bins[2]:
                second_age_range_df = pd.concat([second_age_range_df, subject_df])
            elif self.age_bins[2] <= age <= self.age_bins[3]:
                third_age_range_df = pd.concat([third_age_range_df, subject_df])
        
        first_age_median_df, first_age_std_df = self._get_median_std(first_age_range_df)
        second_age_median_df, second_age_std_df = self._get_median_std(second_age_range_df)
        third_age_median_df, third_age_std_df = self._get_median_std(third_age_range_df)

        return [first_age_median_df, second_age_median_df, third_age_median_df], \
               [first_age_std_df, second_age_std_df, third_age_std_df], \
               [first_age_range_df, second_age_range_df, third_age_range_df]

    def get_axial_statistics(
            self,
            subjects_dirs: SubjectsDict
            ) -> StatisticsResult:
        """
        Separate and calculate statistics for subjects based on their axial length.

        :param subjects_dirs: Dictionary with patient IDs as keys and tuples of parameters as values.
        :type subjects_dirs: Dict[str, Tuple[int, float, float, str, datetime, pd.DataFrame]]
        :return: Tuple containing pd.DataFrames for medians and standard deviations of different axial length ranges.
        :rtype: Tuple[pd.DataFrame]
        """
        # Initialize pd.DataFrames for different axial length ranges
        first_axial_range_df = pd.DataFrame()
        second_axial_range_df = pd.DataFrame()
        third_axial_range_df = pd.DataFrame()
        fourth_axial_range_df = pd.DataFrame()

        # Separate axials for plot
        for subject_id in subjects_dirs:
            _, axial, _, _, _, subject_df = subjects_dirs[subject_id]
            if self.axial_length_bins[0] <= axial < self.axial_length_bins[1]:
                first_axial_range_df = pd.concat([first_axial_range_df, subject_df])
            elif self.axial_length_bins[1] <= axial < self.axial_length_bins[2]:
                second_axial_range_df = pd.concat([second_axial_range_df, subject_df])
            elif self.axial_length_bins[2] <= axial < self.axial_length_bins[3]:
                third_axial_range_df = pd.concat([third_axial_range_df, subject_df])
            elif self.axial_length_bins[3] <= axial <= self.axial_length_bins[4]:
                fourth_axial_range_df = pd.concat([fourth_axial_range_df, subject_df])

        # Get medians and standard deviations
        first_axial_median_df, first_axial_std_df = self._get_median_std(first_axial_range_df)
        second_axial_median_df, second_axial_std_df = self._get_median_std(second_axial_range_df)
        third_axial_median_df, third_axial_std_df = self._get_median_std(third_axial_range_df)
        fourth_axial_median_df, fourth_axial_std_df = self._get_median_std(fourth_axial_range_df)
        return [first_axial_median_df, second_axial_median_df, third_axial_median_df, fourth_axial_median_df], \
               [first_axial_std_df, second_axial_std_df, third_axial_std_df, fourth_axial_std_df], \
               [first_axial_range_df, second_axial_range_df, third_axial_range_df, fourth_axial_range_df]

    def get_spherical_statistics(
        self,
        subjects_dirs: SubjectsDict
        ) -> StatisticsResult:
        """
        Separate and calculate statistics for subjects based on their spherical equivalence parameter.

        :param subjects_dirs: Dictionary with patient IDs as keys and tuples of parameters as values.
        :type subjects_dirs: Dict[str, Tuple[int, float, float, str, datetime, pd.DataFrame]]
        :return: Tuple containing pd.DataFrames for medians and standard deviations of different spherical equivalence ranges.
        :rtype: Tuple[pd.DataFrame]
        """
        first_spherical_range_df = pd.DataFrame()
        second_spherical_range_df = pd.DataFrame()
        third_spherical_range_df = pd.DataFrame()

        # Separate sphericals for plot
        for subject_id in subjects_dirs:
            _, _, spherical, _, _, subject_df = subjects_dirs[subject_id]
            if self.spherical_equiv_bins[0][0] <= spherical <= self.spherical_equiv_bins[0][1]:
                first_spherical_range_df = pd.concat([first_spherical_range_df, subject_df])
            elif self.spherical_equiv_bins[1][0] <= spherical <= self.spherical_equiv_bins[1][1]:
                second_spherical_range_df = pd.concat([second_spherical_range_df, subject_df])
            elif self.spherical_equiv_bins[2][0] <= spherical <= self.spherical_equiv_bins[2][1]:
                third_spherical_range_df = pd.concat([third_spherical_range_df, subject_df])

        # Get the median & std
        first_spherical_median_df, first_spherical_std_df = self._get_median_std(first_spherical_range_df)
        second_spherical_median_df, second_spherical_std_df = self._get_median_std(second_spherical_range_df)
        third_spherical_median_df, third_spherical_std_df = self._get_median_std(third_spherical_range_df)

        return [first_spherical_median_df, second_spherical_median_df, third_spherical_median_df], \
               [first_spherical_std_df, second_spherical_std_df, third_spherical_std_df], \
               [first_spherical_range_df, second_spherical_range_df, third_spherical_range_df]

    def get_sex_statistics(
        self,
        subjects_dirs: SubjectsDict
        ) -> StatisticsResult:
        """
        Separate and calculate statistics for subjects based on their sex.

        :param subjects_dirs: Dictionary with patient IDs as keys and tuples of parameters as values.
        :type subjects_dirs: Dict[str, Tuple[int, float, float, str, datetime, pd.DataFrame]]
        :return: Tuple containing pd.DataFrames for medians and standard deviations of male and female subjects.
        :rtype: Tuple[pd.DataFrame]
        """
        m_range_df = pd.DataFrame()
        f_range_df = pd.DataFrame()

        # Separate sex for plot
        for subject_id in subjects_dirs:
            _, _, _, sex, _, subject_df = subjects_dirs[subject_id]
            if sex == 'H':
                m_range_df = pd.concat([m_range_df, subject_df])
            elif sex == 'F':
                f_range_df = pd.concat([f_range_df, subject_df])

        # Get the medians & stds
        m_median_df, m_std_df = self._get_median_std(m_range_df)
        f_median_df, f_std_df = self._get_median_std(f_range_df)
        return [m_median_df, f_median_df], [m_std_df, f_std_df], [m_range_df, f_range_df]

    def get_time_visits_statistics(
        self,
        subjects_dirs: SubjectsDict
        ) -> StatisticsResult:
        """
        Separate and calculate statistics for subjects based on their visit dates.

        :param subjects_dirs: Dictionary with patient IDs as keys and tuples of parameters as values.
        :type subjects_dirs: Dict[str, Tuple[int, float, float, str, datetime, pd.DataFrame]]
        :return: Tuple containing pd.DataFrames for medians and standard deviations of different time periods.
        :rtype: Tuple[pd.DataFrame]
        """
        first_part_21_range_df = pd.DataFrame()
        second_part_21_range_df = pd.DataFrame()
        first_part_22_range_df = pd.DataFrame()

        # Separate dates of visit for plot
        for subject_id in subjects_dirs:
            _, _, _, _, date_visit, subject_df = subjects_dirs[subject_id]
            if date_visit < self.visit_dates_bins[0]:
                first_part_21_range_df = pd.concat([first_part_21_range_df, subject_df])
            elif self.visit_dates_bins[0] <= date_visit <= self.visit_dates_bins[1]:
                second_part_21_range_df = pd.concat([second_part_21_range_df, subject_df])
            elif date_visit > self.visit_dates_bins[1]:
                first_part_22_range_df = pd.concat([first_part_22_range_df, subject_df])

        # Gather the medians of the groups
        first_part_21_median_df, first_part_21_std_df = self._get_median_std(first_part_21_range_df)
        second_part_21_median_df, second_part_21_std_df = self._get_median_std(second_part_21_range_df)
        first_part_22_median_df, first_part_22_std_df = self._get_median_std(first_part_22_range_df)
        return [first_part_21_median_df, second_part_21_median_df, first_part_22_median_df], \
               [first_part_21_std_df, second_part_21_std_df, first_part_22_std_df], \
               [first_part_21_range_df, second_part_21_range_df, first_part_22_range_df]

if __name__ == '__main__':
    output_path = Path(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\all_stats_new')
    Parser.initialize()
    # BaselineCharacteristics(output_path, 0.1).compare_to_baseline_characteristics()
    BC = BaselineCharacteristics(output_path, 0.1)
    healthy_data = BC.get_patient_data()
    subjects_dirs = BC.gather_parameters(healthy_data)
    BC.initialize_parameters()

    BaselineCharacteristicsPlotter(
        BC.output_path, BC.min_ecc, BC.max_ecc, BC.eccentricity_bins
    ).plot_subject_against_subject(subjects_dirs)