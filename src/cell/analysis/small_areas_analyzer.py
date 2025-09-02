import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_checkpoint


class TooSmallAreasAnalyzer:
    """
    A class to analyze areas that are too small and lead to high cone densities.

    Methods
    -------
    run():
        Executes the analysis for too small areas.
    too_small_areas_analysis() -> None:
        Analyzes the too small areas that lead to high cone densities.
    """
    def __init__(self):
        """
        Initializes the TooSmallAreasAnalyzer class.
        """
        pass

    def run(self):
        """
        Executes the analysis for too small areas.

        This method runs the analysis for too small areas and generates scatter plots.
        """
        self.too_small_areas_analysis()
        self.too_small_areas_analysis_scatter()

    def too_small_areas_analysis(self) -> None:
        """
        Analyzes the too small areas that lead to high cone densities.

        This method gathers the results from patient folders and creates box plots
        to visualize the number of pixels in regions when computing density.
        """
        # gather the results in patient's folders
        too_big_areas_total = []
        normal_areas_total = []
        subject_dirs = [file for file in os.listdir(self.dirs_to_process) if "Subject"
                        in file and os.path.isdir(os.path.join(self.dirs_to_process, file))]
        for subject_dir in subject_dirs:
            subject_path = os.path.join(self.dirs_to_process, subject_dir)
            session_dirs = [file for file in os.listdir(subject_path) if "Session"
                            in file and os.path.isdir(os.path.join(subject_path, file))]
            for session_dir in session_dirs:
                session_path = os.path.join(subject_path, session_dir)
                test_path = os.path.join(session_path, self.test_dir)
                try:
                    too_big_areas, normal_areas = load_checkpoint(test_path,
                                                                  "too_big_areas",
                                                                  "normal_areas")
                except ValueError:
                    continue
                too_big_areas_total.extend(too_big_areas)
                normal_areas_total.extend(normal_areas)

        # create the plots
        plt.boxplot([too_big_areas_total, normal_areas_total],
                    showfliers=False)
        plt.title(
            "Number of pixels in a region when computing density", fontsize=20)
        plt.tick_params(labelsize=14)
        plt.ylabel("Number of pixels", fontsize=18)
        plt.xticks([1, 2], ["Density>100'000", "Density<100'000"], fontsize=18)
        plt.savefig(os.path.join(self.out_path, "too_big_no_outliers.png"))
        plt.close()

        plt.boxplot([too_big_areas_total, normal_areas_total])
        plt.title(
            "Number of pixels in a region when computing density", fontsize=20)
        plt.tick_params(labelsize=14)
        plt.ylabel("Number of pixels", fontsize=18)
        plt.xticks([1, 2], ["Density>100'000", "Density<100'000"], fontsize=18)
        plt.savefig(os.path.join(self.out_path, "too_big.png"))
        plt.close()

    def too_small_areas_analysis_scatter(self) -> None:
        """
        Analyze the areas that were too small in number of pixels and that were mostly
        giving the abnormally high cone densities

        This function helped to choose the threshold (4819) that we select to discard
        too small areas in density computations
        """

        # Gather the resulting areas from each of the subjects
        areas_total = []
        subject_dirs = [file for file in os.listdir(self.dirs_to_process) if "Subject"
                        in file and os.path.isdir(os.path.join(self.dirs_to_process, file))]
        for subject_dir in subject_dirs:
            subject_path = os.path.join(self.dirs_to_process, subject_dir)
            session_dirs = [file for file in os.listdir(subject_path) if "Session"
                            in file and os.path.isdir(os.path.join(subject_path, file))]
            for session_dir in session_dirs:
                session_path = os.path.join(subject_path, session_dir)
                test_path = os.path.join(session_path, self.test_dir)
                try:
                    areas = load_checkpoint(test_path, "areas")
                except ValueError:
                    continue
                try:
                    areas_total.extend(areas[0])
                except IndexError:
                    continue

        # plot the areas vs the cone density found
        threshold = 4819
        plt.scatter(np.array(areas_total)[:, 1], np.array(areas_total)[:, 0])
        plt.title("Number of pixels in a region compared to density", fontsize=20)
        plt.xlabel("Number of pixels", fontsize=18)
        plt.yscale("log")
        plt.axvline(x=threshold, color='b')
        plt.ylabel("Ln of Cone Density [Cells/mm^2]", fontsize=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig(os.path.join(self.out_path, "too_big_scatter.png"))
        plt.close()

        # we threshold cone densities that are too big (>100'000) to be our Negatives
        # so that we can compute a score based on the densities
        too_big_areas_total = []
        normal_areas_total = []
        for cone in areas_total:
            if cone[0] > 100000:
                too_big_areas_total.append(cone)
            else:
                normal_areas_total.append(cone)
        plt.scatter(np.array(too_big_areas_total)[:, 1], np.array(too_big_areas_total)[:, 0],
                    color='red', label='Density > 100000')
        plt.scatter(np.array(normal_areas_total)[:, 1], np.array(normal_areas_total)[:, 0],
                    color='blue', label='Density < 100000')
        plt.title("Number of pixels in a region compared to density", fontsize=20)
        plt.xlabel("Number of pixels", fontsize=18)
        plt.yscale("log")
        plt.ylabel("Ln of Cone Density [Cells/mm^2]", fontsize=18)
        plt.axvline(x=threshold, color='b')
        plt.legend(fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig(os.path.join(self.out_path,
                    "too_big_scatter_separated.png"))
        plt.close()

        # construct False Positives, False Negatives, True Positives and True Negatives
        # for our analysis
        TP = []
        FN = []
        FP = []
        TN = []
        for cone in areas_total:
            if cone[0] > 100000:
                if cone[1] < threshold:
                    TP.append(cone)
                else:
                    FP.append(cone)
            else:
                if cone[1] < threshold:
                    FN.append(cone)
                else:
                    TN.append(cone)

        # plot the results with color code
        plt.scatter(np.array(TP)[:, 1], np.array(TP)[:, 0],
                    color='green', label='TP')
        plt.scatter(np.array(TN)[:, 1], np.array(TN)[:, 0],
                    color='black', label='TN')
        plt.scatter(np.array(FP)[:, 1], np.array(FP)[:, 0],
                    color='red', label='FP')
        plt.scatter(np.array(FN)[:, 1], np.array(FN)[:, 0],
                    color='blue', label='FN')
        plt.title("Number of pixels in a region compared to density", fontsize=20)
        plt.xlabel("Number of pixels", fontsize=18)
        plt.yscale("log")
        plt.ylabel("Ln of Cone Density [Cells/mm^2]", fontsize=18)
        plt.axvline(x=threshold, color='b')
        plt.legend(fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig(os.path.join(self.out_path,
                    "too_big_scatter_separated_and_thresholded.png"))
        plt.close()

        # print the results
        print(
            f"Accuracy: {(len(TP)+len(TN))/(len(TP)+len(FN)+len(FP)+len(TN))}")
        print(f"Precision: {(len(TP))/(len(TP)+len(FP))}")
        print(f"Recall: {(len(TP))/(len(TP)+len(FN))}")
        print(f"F1: {2*((len(TP))/(len(TP)+len(FP))*(len(TP))/(len(TP)+len(FN)))/((len(TP))/(len(TP)+len(FP))+(len(TP))/(len(TP)+len(FN)))}")

        # mean: 4819:
        # Accuracy: 0.9907639530281039
        # Precision: 0.6761904761904762
        # Recall: 0.6635514018691588
        # f1 : 0.669811320754717

        # median: 1634:
        # Accuracy: 0.9914236706689536
        # Precision: 0.47619047619047616
        # Recall: 0.8333333333333334
        # F1:0.6060606060606061
