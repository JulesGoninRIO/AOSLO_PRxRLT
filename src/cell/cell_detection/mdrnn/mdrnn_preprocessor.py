import os
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import re

from src.cell.cell_detection.mdrnn.mdrnn_preprocessing_params import MDRNNPreProcessingParams
from src.shared.datafile.datafile_constants import ImageModalities
from src.shared.datafile.image_file import ImageFile
from src.cell.cell_detection.nst.nst_pipeline_manager import NSTPipelineManager
from src.cell.cell_detection.patch_cropper import PatchCropper
from src.cell.cell_detection.corresponding_patch_finder import CorrespondingPatchFinder
from src.cell.montage.montage_mosaic import MontageMosaic
from src.cell.processing_path_manager import ProcessingPathManager

def match_histograms(src_hist: np.ndarray, ref_hist: np.ndarray) -> np.ndarray:
    """
    This method matches the source image histogram to the reference signal

    :param src_hist: The original source image
    :type src_hist: np.ndarray
    :param ref_hist: The reference image
    :type ref_hist: np.ndarray
    :return: image_after_matching
    :rtype: np.ndarray
    """

    # Compute the normalized cdf for the source and reference image
    src_cdf = calculate_cdf(src_hist)
    ref_cdf = calculate_cdf(ref_hist)

    # Make a separate lookup table for each color
    lookup_table = calculate_lookup(src_cdf, ref_cdf)

    return lookup_table


def calculate_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    This method calculates the cumulative distribution function

    :param histogram: the values of the histogram
    :type histogram: np.ndarray
    :return: the normalized cumulative distribution function
    :rtype: np.ndarray
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """
    This method creates the lookup table

    :param src_cdf: The cdf for the source image
    :type src_cdf: np.ndarray
    :param ref_cdf: The cdf for the reference image
    :type ref_cdf: np.ndarray
    :return: The lookup table
    :rtype: np.ndarray
    """

    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val

    return lookup_table

def plot_legends(plot: plt,
                 legends: dict = None,
                 hist: np.ndarray = None,
                 vline: bool = False,
                 pixels_before: np.ndarray = None) -> plt:
    """
    Helper to plot legend to histogram

    :param plot: the plot to add leged to
    :type plot: plt
    :param legends: the legend to add, defaults to None
    :type legends: dict, optional
    :param hist: the histogram to add to the plot, defaults to None
    :type hist: np.array, optional
    :param vline: the vertical line to add, defaults to False
    :type vline: bool, optional
    :param pixels_before: the pixels before to add, defaults to None
    :type pixels_before: np.array, optional
    :return: the updated matplotlib.pyplot plot
    :rtype: plt
    """

    if vline:
        if pixels_before:
            hist = sum(np.asarray(
                list(pixels_before.values()), dtype=object)[:, 0])
        bins = np.linspace(0, 256, 257)
        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=hist)
        var = np.average((mids - mean)**2, weights=hist)
        max_hist = np.max(hist)
        plt.vlines(x=[mean-np.sqrt(var), mean+np.sqrt(var)],
                   ymin=0, ymax=max_hist, label="std", color="r")
        plt.vlines(x=[mean-var, mean+var], ymin=0,
                   ymax=max_hist, label="var", color="b")
        plt.vlines(x=mean, ymin=0, ymax=max_hist, label="mean", color="g")
        plt.legend()
    if "title" in legends.keys():
        plt.title(legends["title"], fontsize=14)
    if "xlabel" in legends.keys():
        plt.xlabel(legends["xlabel"], fontsize=14)
    if "ylabel" in legends.keys():
        plt.ylabel(legends["ylabel"], fontsize=14)
    if "write_legend" in legends.keys():
        plt.legend(prop={'size': 14})

    return plt

def plot_multiple_histogram(output_dir: str,
                            hists: dict,
                            legends: dict = None,
                            vline: bool = False) -> None:
    """
    Plot multiple histograms from data contained in hists

    :param output_dir: the directory where to save the files
    :type output_dir: str
    :param hists: the histograms to plot the data
    :type hists: dict
    :param legends: the legend of the graph if any, defaults to None
    :type legends: dict, optional
    :param vline: the pixels before to plot also, defaults to False
    :type vline: bool, optional
    """

    fig = plt.figure()
    for key, val in hists.items():
        plt.plot(val, label=key)

    plot = plot_legends(plt, legends, val, vline)

    plot.xticks(fontsize=14)
    plot.yticks(fontsize=14)
    if "name" in legends.keys():
        plot.savefig(os.path.join(output_dir, legends["name"]+".png"))
    else:
        plot.savefig(os.path.join(output_dir, "no_name.png"))

    plot.close(fig)

def hist_matching(bad_image: np.ndarray,
                  good_hist: np.ndarray,
                  output_dir: str = None,
                  image_name: str = None) -> np.ndarray:
    """
    Method to match histograms from bad patches to good one. Bad means few number
    of cones detected by the MDRNN algorithm

    :param bad_image: the bad images that have to be matches
    :type bad_image: np.ndarray
    :param good_hist: the good patch to match the bad one to
    :type good_hist: np.ndarray
    :param output_dir: the output directory to save matches, defaults to None
    :type output_dir: str, optional
    :param image_name: the name of the image, defaults to None
    :type image_name: str, optional
    :return: the updated bad patch that has been matches
    :rtype: np.ndarray
    """

    # Look for output_dir and choose right one
    # fig_dir = Path(output_dir).parent.absolute()
    fig_dir = os.path.join(output_dir, "hist_matching")
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    bad_hist = np.histogram(bad_image.flatten(), 256, [0, 256])

    lookup_table = match_histograms(bad_hist, good_hist)
    # Use the lookup function to transform the colors of the original
    # source image
    after_transform = cv2.LUT(bad_image, lookup_table)

    # Put the image back together
    # image_after_matching = cv2.merge([
    #    blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(after_transform)

    match_hist = np.histogram(image_after_matching.flatten(), 256, [0, 256])
    hists = {'Not Working histogram': bad_hist[0],
             'Working histogram': good_hist, 'Matched histogram': match_hist}
    legends = {'xlabel': "Image pixels [0:255]",
               'ylabel': "Number of pixels",
               'name': f"histogram_matching_{image_name}",
               "write_legend": True}
    plot_multiple_histogram(fig_dir, hists, legends)

    return image_after_matching

def preprocess_images(images):
    def replace_zero_with_mean(image):
        rows, cols = image.shape
        new_image = image.copy()
        for i in range(rows):
            for j in range(cols):
                if image[i, j] == 0:
                    neighbors = []
                    if i > 0 and image[i-1, j] != 0:  # Up
                        neighbors.append(image[i-1, j])
                    if i < rows-1 and image[i+1, j] != 0:  # Down
                        neighbors.append(image[i+1, j])
                    if j > 0 and image[i, j-1] != 0:  # Left
                        neighbors.append(image[i, j-1])
                    if j < cols-1 and image[i, j+1] != 0:  # Right
                        neighbors.append(image[i, j+1])
                    if neighbors:
                        new_image[i, j] = np.mean(neighbors)
        return new_image
    return [replace_zero_with_mean(image) for image in images]

class MDRNNPreProcessor():
    """
    Pre-processor for MDRNN.

    This class handles the pre-processing of images for the MDRNN algorithm, including cropping images
    into patches and applying pre-processing techniques.

    :param path_manager: The path manager for handling file paths.
    :type path_manager: ProcessingPathManager
    :param montage_mosaic: The montage mosaic to use for pre-processing, defaults to None.
    :type montage_mosaic: optional
    :param preprocessing_params: The parameters for pre-processing, defaults to None.
    :type preprocessing_params: MDRNNPreProcessingParams, optional
    """
    def __init__(
        self,
        path_manager: ProcessingPathManager,
        montage_mosaic: MontageMosaic = None,
        preprocessing_params: MDRNNPreProcessingParams=None):
        self.path_manager = path_manager
        self.montage_mosaic = montage_mosaic
        self.preprocessing_params = preprocessing_params
        self.preprocessed_patches = []
        # self.__raw_run_path = ""
        # self.__base_dir = self.__raw_run_path.parent
        # self.__output_dir = self.__base_dir / 'var_border_both_global'
        # self.__postprocess_dir = self.__base_dir / 'postprocessed'
        # self.__corrected_montage_path = self.__base_dir / 'montaged_corrected'
        if self.preprocessing_params is not None:
            self.curring_hist = self.compute_curring_histogram()

    def run(self) -> List[ImageFile]:
        """
        Run the pre-processing steps.

        This method runs the pre-processing steps, including cropping images into patches and
        applying pre-processing techniques to the images.

        :return: A list of pre-processed patches.
        :rtype: List[ImageFile]
        """
        all_patches = self.crop_images_to_patches()
        self.apply_preprocessing_to_images(all_patches)
        return self.preprocessed_patches  # [str(patch.crop_id) for patch in all_patches]

    def crop_images_to_patches(self) -> List[ImageFile]:
        """
        Crop images into patches.

        This method crops the images into patches and applies initial pre-processing to the patches.

        :return: A list of cropped image patches.
        :rtype: List[ImageFile]
        """
        all_patches = []
        counter = 0
        # for file in self.path_manager.path.iterdir():
        for file in os.scandir(self.path_manager.path):
            if not ImageModalities.CS.value in file.name:
                continue
            try:
                image = ImageFile(file.name)
            except ValueError:
                continue
            image.read_data(self.path_manager.path)
            image.data = preprocess_images([image.data])[0]
            if image.prefix:
                self.preprocess_patch(image)
                # TODO: verify it is indeed needed to rewrite data
                # but I think it used to be because of image type, might be not
                # the case anymore
                all_patches.append(image)
            else:
                patch_cropper = PatchCropper(image)
                patches, counter = patch_cropper.get_patches(counter)
                all_patches.extend(patches)
        return all_patches

    def apply_preprocessing_to_images(self, all_patches: List[ImageFile]):
        """
        Apply pre-processing to images.

        This method applies the specified pre-processing techniques to the list of image patches.

        :param all_patches: The list of image patches to pre-process.
        :type all_patches: List[ImageFile]
        :return: None
        """
        if self.preprocessing_params is not None:
            if self.preprocessing_params.enhancement == "match" or self.preprocessing_params.enhancement == "nst":
                previous_name = str(self.preprocessing_params)
                previous_dir = re.sub("_" + self.preprocessing_params.enhancement, "", previous_name)
                self.previous_path = self.path_manager.path / previous_dir
                histogram_per_patch = pixel_histogram_per_patch(self.previous_path)
                corr_patch = CorrespondingPatchFinder(self.montage_mosaic, self.previous_path, histogram_per_patch).get_corresponding_patches()
            else:
                corr_patch = None
        else:
            corr_patch = None
        for patch in all_patches:
            self.preprocess_patch(patch, corr_patch)

    def preprocess_patch(self, patch: ImageFile, corr_patch: Dict[ImageFile, ImageFile]):
        """
        Preprocess the patches regarding the parameters stated.

        This method preprocesses the given image patch according to the specified pre-processing parameters.
        It applies various pre-processing techniques such as range processing, zero method, histogram matching,
        and neural style transfer (NST).

        :param patch: The ImageFile to preprocess.
        :type patch: ImageFile
        :param corr_patch: A dictionary mapping image patches to their corresponding patches for processing.
        :type corr_patch: Dict[ImageFile, ImageFile]
        :return: None
        """
        # we want to correct output for further run (either zero, var, or only std)
        if self.preprocessing_params is not None:
            if self.preprocessing_params.method:
                if not self.preprocessing_params.method == "zero":
                    if self.preprocessing_params.range_method == "lower":
                        self.process_lower_range(patch.data)
                    elif self.preprocessing_params.range_method == "both":
                        self.process_both_ranges(patch.data)
                else:
                    self.process_zero_method(patch.data)
            if self.preprocessing_params.enhancement == "match":
                self.apply_histogram_matching_to_patches(patch, corr_patch)
            if self.preprocessing_params.enhancement:
                # TODO: maybe add support for histogram matching, CLAHE or others
                self.apply_nst_to_patches(patch, corr_patch)
            else:
                if corr_patch:
                    if patch in corr_patch:
                        patch.write_data(self.path_manager.mdrnn.output_path)
                else:
                    patch.write_data(self.path_manager.mdrnn.output_path)
        else:
            patch.write_data(self.path_manager.mdrnn.output_path)
        self.preprocessed_patches.append(str(patch.crop_id))

    def apply_histogram_matching_to_patches(self, corr_patch: Dict[ImageFile, ImageFile]):
        """
        Apply histogram matching to patches.

        This method applies histogram matching to the given image patches based on the corresponding patches.

        :param corr_patch: A dictionary mapping image patches to their corresponding patches for histogram matching.
        :type corr_patch: Dict[ImageFile, ImageFile]
        :return: None
        """
        name = None
        for key, val in list(corr_patch)[0].items():
            if name == key:
                cropped = hist_matching(
                    cropped, self.histogram_per_patch[val], self.path_manager.mdrnn.output_path, name)
                break

    def apply_nst_to_patches(self, patch: ImageFile, corr_patch: Dict[ImageFile, ImageFile]):
        """
        Apply neural style transfer (NST) to patches.

        This method applies neural style transfer to the given image patch based on the corresponding patch.

        :param patch: The ImageFile to apply NST to.
        :type patch: ImageFile
        :param corr_patch: A dictionary mapping image patches to their corresponding patches for NST.
        :type corr_patch: Dict[ImageFile, ImageFile]
        :return: None
        """
        if patch in corr_patch:
            style_image_file = corr_patch[patch]
            print("Runnin gon style Image file: ", style_image_file)
            style_image_file.read_data(self.previous_path)
            patch.read_data(self.previous_path)
            nst_pipeline_manager = NSTPipelineManager(style_image_file, patch)  # run_nst(val, key
            cropped = nst_pipeline_manager.run()
            nst_pipeline_manager.generate_plot(self.path_manager.mdrnn.output_path)
            nst_pipeline_manager.save_losses(self.path_manager.mdrnn.output_path)
            # print("oe")
            # TODO: save image
            # run_nst(val, key, self.output_path, self.previous_path)
            patch_name = self.path_manager.mdrnn.output_path / str(patch)
            # # patch.write_data(patch_name)
            # cv2.imwrite(patch_name, cropped)
            # from torchvision.utils import save_image
            save_image(cropped, str(patch_name))
        else:
            patch.write_data(self.path_manager.mdrnn.output_path)

    def process_lower_range(self, patch_data: np.ndarray):
        """
        Process the lower range of the patch data.

        This method processes the lower range of the patch data based on the specified replacement strategy.

        :param patch_data: The patch data to process.
        :type patch_data: np.ndarray
        :raises ValueError: If the replace parameter is not 'median' or 'border'.
        :return: None
        """
        if self.preprocessing_params.replace == "median":
            self.process_median_replace(patch_data, lower=True)
        elif self.preprocessing_params.replace == "border":
            self.process_border_replace(patch_data, lower=True)
        else:
            raise ValueError("replace parameter must be either 'median' or 'border'")

    def process_both_ranges(self, patch: ImageFile):
        """
        Process both the lower and upper ranges of the patch data.

        This method processes both the lower and upper ranges of the patch data based on the specified replacement strategy.

        :param patch: The ImageFile containing the patch data to process.
        :type patch: ImageFile
        :raises ValueError: If the replace parameter is not 'median' or 'border'.
        :return: None
        """
        if self.preprocessing_params.replace == "median":
            self.process_median_replace(patch, lower=True, upper=True)
        elif self.preprocessing_params.replace == "border":
            self.process_border_replace(patch, lower=True, upper=True)
        else:
            raise ValueError("replace parameter must be either 'median' or 'border'")

    def process_zero_method(self, patch_data: np.ndarray):
        """
        Process the patch data using the zero method.

        This method processes the patch data using the zero method based on the specified replacement strategy.

        :param patch_data: The patch data to process.
        :type patch_data: np.ndarray
        :raises ValueError: If the replace parameter is not 'median' or 'border'.
        :return: None
        """
        if self.preprocessing_params.replace == "median":
            self.process_median_replace(patch_data, zero=True)
        elif self.preprocessing_params.replace == "border":
            self.process_border_replace(patch_data, zero=True)
        else:
            raise ValueError("replace parameter must be either 'median' or 'border'")

    def process_median_replace(self, patch_data: np.array, lower: bool = False, upper: bool = False, zero: bool = False):
        """
        Process the patch data using median replacement.

        This method processes the patch data by replacing values with the median based on the specified conditions
        (lower, upper, zero) and correction method (local or global).

        :param patch_data: The patch data to process.
        :type patch_data: np.array
        :param lower: Whether to process the lower range of the data, defaults to False.
        :type lower: bool, optional
        :param upper: Whether to process the upper range of the data, defaults to False.
        :type upper: bool, optional
        :param zero: Whether to process zero values in the data, defaults to False.
        :type zero: bool, optional
        :raises ValueError: If the correct parameter is not 'local' or 'global'.
        :return: None
        """
        if self.preprocessing_params.correct == "local":
            if self.preprocessing_params.method == "var":
                if lower:
                    patch_data[patch_data <= np.mean(patch_data)-np.var(patch_data)] = np.median(patch_data)
                if upper:
                    patch_data[patch_data >= np.mean(patch_data)+np.var(patch_data)] = np.median(patch_data)
                if zero:
                    patch_data[patch_data <= 0] = np.median(patch_data)
            elif self.preprocessing_params.method == "std":
                if lower:
                    patch_data[patch_data <= np.mean(patch_data)-np.std(patch_data)] = np.median(patch_data)
                if upper:
                    patch_data[patch_data >= np.mean(patch_data)+np.std(patch_data)] = np.median(patch_data)
        elif self.preprocessing_params.correct == "global":
            if lower:
                patch_data[patch_data <= self.curring_hist[0]-self.curring_hist[1]] = self.curring_hist[2]
            if upper:
                patch_data[patch_data >= self.curring_hist[0]+self.curring_hist[1]] = self.curring_hist[2]
            if zero:
                patch_data[patch_data <= 0] = self.curring_hist[2]
        else:
            raise ValueError("correct parameter must be either 'local' or 'global'")

    def process_border_replace(self, patch_data: np.array, lower: bool = False, upper: bool = False, zero: bool = False):
        """
        Process the patch data using border replacement.

        This method processes the patch data by replacing values with the border values based on the specified conditions
        (lower, upper, zero) and correction method (local or global).

        :param patch_data: The patch data to process.
        :type patch_data: np.array
        :param lower: Whether to process the lower range of the data, defaults to False.
        :type lower: bool, optional
        :param upper: Whether to process the upper range of the data, defaults to False.
        :type upper: bool, optional
        :param zero: Whether to process zero values in the data, defaults to False.
        :type zero: bool, optional
        :raises ValueError: If the correct parameter is not 'local' or 'global'.
        :return: None
        """
        if self.preprocessing_params.correct == "local":
            if self.preprocessing_params.method == "var":
                if lower:
                    patch_data[patch_data <= np.mean(patch_data)-np.var(patch_data)] = np.mean(patch_data)-np.var(patch_data)
                if upper:
                    patch_data[patch_data >= np.mean(patch_data)+np.var(patch_data)] = np.mean(patch_data)+np.var(patch_data)
                if zero:
                    patch_data[patch_data <= 0] = np.mean(patch_data)-np.var(patch_data)
            elif self.preprocessing_params.method == "std":
                if lower:
                    patch_data[patch_data <= np.mean(patch_data)-np.std(patch_data)] = np.mean(patch_data)-np.std(patch_data)
                if upper:
                    patch_data[patch_data >= np.mean(patch_data)+np.std(patch_data)] = np.mean(patch_data)+np.std(patch_data)
                if zero:
                    patch_data[patch_data <= 0] = np.mean(patch_data)-np.std(patch_data)
        elif self.preprocessing_params.correct == "global":
            if lower:
                patch_data[patch_data <= self.curring_hist[0]-self.curring_hist[1]] = self.curring_hist[0]-self.curring_hist[1]
            if upper:
                patch_data[patch_data >= self.curring_hist[0]+self.curring_hist[1]] = self.curring_hist[0]+self.curring_hist[1]
            if zero:
                patch_data[patch_data <= 0] = self.curring_hist[0]-self.curring_hist[1]
        else:
            raise ValueError("correct parameter must be either 'local' or 'global'")

    def compute_curring_histogram(self) -> Tuple[float, float, float]:
        """
        Compute the curring histogram.

        This method computes the curring histogram statistics based on the pixel histogram per patch.

        :return: A tuple containing the mean, standard deviation, and median of the histogram.
        :rtype: Tuple[float, float, float]
        """
        histogram_per_patch = pixel_histogram_per_patch(self.path_manager.mdrnn.first_run_path)
        curring_hist = get_histograms_statistics(self.preprocessing_params.method, list(histogram_per_patch.values()))
        return curring_hist

def pixel_histogram_per_patch(patch_dir: Path) -> Dict[ImageFile, np.array]: 
    """
    Get the pixel values of every patch used.

    This function retrieves the pixel values of every patch used in Davidson's algorithm.

    :param patch_dir: The directory containing the patches.
    :type patch_dir: Path
    :return: A dictionary with the pixel values for each patch.
    :rtype: Dict[ImageFile, np.array]
    """
    histogram_per_patch = {}
    for file in patch_dir.iterdir():
        if file.is_file() and file.name.endswith(".tif"):
            image = ImageFile(file.name)
            image.read_data(str(patch_dir))
            hist, _ = np.histogram(image.data, 256, [0, 256])
            histogram_per_patch[str(image)] = hist

    return histogram_per_patch

def weighted_median(data: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the median of an array with weights.

    This function computes the median of an array, taking into account the provided weights.

    :param data: The data array.
    :type data: np.ndarray
    :param weights: The weights array.
    :type weights: np.ndarray
    :return: The weighted median of the data.
    :rtype: float
    """
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumulative_weights, cumulative_weights[-1] / 2)
    return sorted_data[median_idx]

def get_histograms_statistics(method: str, histograms: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Get the statistics from Davidson's inputs.

    This function computes the mean, variance (or standard deviation), and median of the pixel histograms.

    :param method: The method to use for computing the statistics ('std' or 'var').
    :type method: str
    :param histograms: The list of pixel histograms.
    :type histograms: List[np.ndarray]
    :return: A tuple containing the mean, variance (or standard deviation), and median.
    :rtype: Tuple[float, float, float]
    """
    val = sum(histograms)
    if np.all(val == 0):
        return (0, 0, 0)
    bins = np.linspace(0, 256, 257)
    mids = 0.5 * (bins[1:] + bins[:-1])
    if mids.shape != val.shape:
        mean = np.average(mids, weights=val, axis=0)
    else:
        mean = np.average(mids, weights=val)
    median = weighted_median(mids, weights=val)
    var = np.average((mids - mean) ** 2, weights=val)
    if method == "std":
        curring_hist = (mean, np.sqrt(var), median)
    elif method == "var":
        curring_hist = (mean, var, median)
    else:
        curring_hist = (mean, 0, 0)
    return curring_hist
