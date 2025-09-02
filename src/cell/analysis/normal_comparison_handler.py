from typing import Tuple
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

from src.configs.parser import Parser
from src.cell.analysis.helpers import gather_results
from src.cell.processing_path_manager import ProcessingPathManager
from src.cell.analysis.constants import RESULT_NAME

NORMAL_DIR = 'compare_to_normal_new'

class NormalComparisonHandler:
    def __init__(self):
        self.step = 0.1

    def compare_to_normal(self, path_manager: ProcessingPathManager):
        self.path_manager = path_manager
        result = self._load_result_data()

        if result is not None:
            stat_dfs = self._gather_results()
            self._process_and_plot_data(result, *stat_dfs, self.path_manager.subject_id)

    def _gather_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df, considered_subj_sess = gather_results(self.path_manager.path.parent.parent, self.path_manager.path)
        assert len(considered_subj_sess) > 1, 'No normal data to compare to.'
        median_df = df.groupby(df.location).apply(lambda x: x.median().where(x.isna().mean(axis=0) < 0.5), include_groups=False)
        # compute as well the 15th and 85th percentiles for shaded plots
        lower_df = df.groupby(df.location).apply(lambda x: x.quantile(0.15).where(x.isna().mean(axis=0) < 0.5), include_groups=False)
        upper_df = df.groupby(df.location).apply(lambda x: x.quantile(0.85).where(x.isna().mean(axis=0) < 0.5), include_groups=False)
        
        return median_df, lower_df, upper_df

    def _load_result_data(self):
        result_path = self.path_manager.density.path / RESULT_NAME
        try:
            return pd.read_csv(result_path, skiprows=1)
        except pd.errors.EmptyDataError:
            return None

    def _process_and_plot_data(self, result_df, median_df, lower_df, upper_df, subject):
        result_df = result_df[abs(result_df.location) <= 10]
        out_path = self.path_manager.path / NORMAL_DIR
        out_path.mkdir(exist_ok=True)

        for column in median_df.columns:
            median = median_df[column].to_numpy()
            lower = lower_df[column].to_numpy()
            upper = upper_df[column].to_numpy()
            result = result_df[['location', column]].to_numpy()
            self._plot_comparison(result, median, lower, upper, column, subject, out_path)

    def _plot_comparison(self, result, median, lower, upper, column, subject, out_path):
        lowess = lambda x,y: sm.nonparametric.lowess(y, x, frac=0.1, xvals=x)
        x = result[:,0]
        if 'densities' in column:
            # no need to smoothen densities again
            result = result[:,1]
        else:
            result = lowess(x, result[:,1])
            median = lowess(x, median)
            lower = lowess(x, lower)
            upper = lowess(x, upper)

        mean_ticks = np.arange(math.floor(min(x)), math.ceil(max(x)) + 1, 1)

        ylabel = 'Cone density [cells/mm²]' if 'densities' in column \
            else 'Choroidal Vascularity Index' if 'CVI' in column \
            else 'Layer thickness [mm]' 

        plt.fill_between(x, lower, upper, color='blue', alpha=0.2, label=r'15th-85th %ile')
        plt.plot(x, median, color='blue', label='Normal')
        plt.plot(x, result, color='red', label=f'Subject {subject}')
        plt.xticks(mean_ticks, mean_ticks)
        plt.xlabel('Retinal eccentricity [°]', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.title(f'Subject {subject} - {re.sub('_', ' ', column)} comparison to Normal', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(str(out_path / f'{re.sub(' ', '_', column)}_compared_to_normal.png'))
        plt.close()


# if __name__ == '__main__':
#     dir_to_process = r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\run\Subject104\Session492'
#     density_analysis_path = r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\run\Subject104\Session492\density_analysis'
#     step = 0.1
#     NormalComparisonHandler(dir_to_process, density_analysis_path, step).compare_to_normal()
