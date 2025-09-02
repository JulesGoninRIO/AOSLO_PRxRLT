from typing import Tuple
from pathlib import Path
from deprecated import deprecated
import json
import svgpathtools
from PIL import Image, ImageDraw
import numpy as np
import scipy.signal
import scipy.ndimage

LAYER_NAMES = [
    'RNFL',
    'GCL+IPL',
    'INL+OPL',
    'ONL',
    'PhotoR+RPE',
    'Choroid',
    'IRF',
    'SRF',
    'PED',
    'CVI',
    'UNKNOWN',
    'OS', # Outer Segment of the Photoreceptors (custom, not in data pulled from Discovery)
]

LAYER_NAMES_TO_INCLUDE = [
    'RNFL',
    'GCL+IPL',
    'INL+OPL',
    'ONL',
    'PhotoR+RPE',
    'Choroid', 
    'OS',
]

def modify_name(layer_name: str) -> str:
    """
    Modify the name of the layer for better comprehension in graphs

    :param layer_name: the layer to rename
    :type layer_name: str
    :return: the new layer name
    :rtype: str
    """
    if layer_name == 'GCL_IPL':
        layer_name = 'GCL+IPL'
    elif layer_name == 'INL_OPL':
        layer_name = 'INL+OPL'
    elif layer_name == 'PR_RPE':
        layer_name = 'PhotoR+RPE'
    elif layer_name == 'CC_CS':
        layer_name = 'Choroid'
    return layer_name

def propagate(arr: list) -> np.array:
    """
    Propagate the array values by filling the None

    :param arr: the array to propagate
    :type arr: np.array
    :return: the array propagated
    :rtype: np.array
    """
    arg_values = np.argwhere(arr)
    for i in range(len(arg_values)):
        if i == len(arg_values)-1:
            for j in range(arg_values[i][0], len(arr)):
                arr[j] = arr[arg_values[i][0]]
        else:
            if i == 0:
                for j in range(arg_values[i][0]):
                    arr[j] = arr[arg_values[i][0]]
            difference_len = arg_values[i+1]-arg_values[i]
            difference_values = arr[arg_values[i+1][0]]-arr[arg_values[i][0]]
            increase_value = difference_values/difference_len
            for j in range(1, difference_len[0]):
                arr[arg_values[i][0] +
                    j] = round(arr[arg_values[i][0]]+j*increase_value[0])
    return arr

def gaussian_filter_nan(data: np.ndarray, sigma: float = 1):
    """
    Apply a Gaussian filter to data, handling NaNs appropriately
    """
    if sigma <= 0:
        return data
    V = data.copy().astype(float)
    V[np.isnan(data)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=sigma)
    W = np.ones_like(data, dtype=float)
    W[np.isnan(data)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=sigma)
    WW[WW == 0] = np.nan
    return VV/WW

def get_cube_center(subject: int | Path, shape: Tuple[int, int] | None = None) -> Tuple[int, int]:
    """
    Get the center of the volume for the given subject (either the subject id
    or the path to its OCT cube dataset), using the white dot in the B-scan for
    which the number is provided in the
    `V:\Studies\AOSLO\data\cohorts\AOSLO healthy\lut_subject_to_oct_data_paths.json` 
    lookup table.
    """
    def __get_center(dataset_path: Path, bscan_id: int) -> Tuple[int, int]:
        dir_oct = dataset_path / 'oct' / 'volume'
        with open(dir_oct / 'info.json') as json_file:
            infos = json.load(json_file)
        dims = {'x': infos['shape'][3], 'y': infos['shape'][0], 'z': infos['shape'][2]}
        seg_file = dataset_path / 'children' / 'segmentation' / f'{bscan_id:04d}.svg'
        oct_file = dir_oct / f'{bscan_id:04d}.jpg'
        oct = Image.open(oct_file)
        assert oct.size == (dims['x'], dims['z'])

        # crop the OCT to the center third, where the white dot is supposed to be
        x_offset = dims['x']//3
        oct = oct.crop((x_offset, 0, dims['x'] - x_offset, dims['z']))

        # extract the segmentation of the RNFL
        svg = svgpathtools.svg2paths(str(seg_file))
        points = [
            (int(line.start.real) - x_offset, int(line.start.imag)) 
            for line in svg[0][1] # RNFL
            if x_offset <= line.start.real <= dims['x'] - x_offset
        ]

        # create a mask of the RNFL using the points from the svg segmentation
        mask = Image.new("L", oct.size, 0)
        oct = np.array(oct)
        oct[oct == 0] = 1  # set 0s to 1s so that true 0s are not confused with the mask
        ImageDraw.Draw(mask).polygon(points, outline=255, fill=255)
        mask = np.array(mask, dtype=bool)

        # restrict the B-scan to a thin centered region of the RNFL
        _max = mask.nonzero()[0].max()
        _min = mask.nonzero()[0].min()
        j = mask[_min:1+_max].sum(axis=0).max()
        oct_pr = (oct * mask)[_max-j:_max+1, :]

        # find the center of the white dot using relative brightness
        center = x_offset + round(np.median(np.nonzero(oct_pr > np.quantile(oct_pr[oct_pr > 0], 0.95))[1]))
        return center, dims['x']

    if isinstance(subject, int):
        subject_id = str(subject)
        with open(r'V:\Studies\AOSLO\data\cohorts\AOSLO healthy\lut_subject_to_oct_data_paths.json', 'r') as f:
            lut_subject_to_paths = json.load(f)
        if subject_id not in lut_subject_to_paths or 'cube' not in lut_subject_to_paths[subject_id]:
            raise ValueError(f'No OCT cube data was found for Subject{subject_id} in the lookup table `V:\\Studies\\AOSLO\\data\\cohorts\\AOSLO healthy\\lut_subject_to_oct_data_paths.json`.\n            Please add the path to the OCT data of the subject in the json file.')
        dataset_path = Path(lut_subject_to_paths[subject_id]['cube'])
    else:
        dataset_path = subject

    with open(r'V:\Studies\AOSLO\data\cohorts\AOSLO healthy\lut_subject_to_white_dot_bscan.json', 'r') as f:
        lut_subject_to_bscan = json.load(f)
    if subject_id not in lut_subject_to_bscan:
        raise ValueError(f'No B-scan was provided for Subject{subject_id} in the lookup table `V:\\Studies\\AOSLO\\data\\cohorts\\AOSLO healthy\\lut_subject_to_white_dot_bscan.json`.\n            Please add the number of the B-scan that contains the white dot, or an educated guess on what the central B-scan is.')
    
    bscan_id = lut_subject_to_bscan[subject_id] - 1 # B-scan numbers on Discovery are 1-indexed
    center, true_bscan_width = __get_center(dataset_path, bscan_id)

    if shape is not None:
        # shape[1] is the width of the layer thichness array (x-axis), which may not
        # be the same as the true width of the B-scan image, because the browse function
        # of CohortExtractor rotates the images a little, so the width may not be the 
        # same. The transformation is linear tho; we can simply use the ratio to recover 
        # where the center should be.
        return bscan_id, round(center * shape[1] / true_bscan_width)
    else:
        return bscan_id, center

@deprecated
def get_center_peak(layer: np.ndarray) -> Tuple[int, int]:
    """
    Get the center peak of the 3d surface defined by the 2-dim array of the layer.

    :param layer: the 3d surface
    :type layer: np.ndarray
    :return: the peak of the layer
    :rtype: Tuple[int, int]
    """

    peak_layers = []
    peaks = []
    for single_layer in layer:
        # only works with 1d arrays
        peak_layers.append(scipy.signal.find_peaks(
            single_layer, distance=1729, prominence=0.1, width=5))
    for peak in peak_layers:
        try:
            new_peak = scipy.signal.find_peaks(
                np.array(layer)[:, peak[0][0]], distance=97, prominence=0.1, width=5)
            try:
                new_peak[0][0]
            except IndexError:
                peaks.append(0)
                continue
            peaks.append(new_peak)
        except IndexError:
            peaks.append(0)

    # Gaussian Filter the layer for the plots
    layer = np.array(layer)
    layer_5 = scipy.ndimage.gaussian_filter(layer, sigma=5)
    pos_layer_5 = layer_5[round(layer_5.shape[0]/3):-round(
        layer_5.shape[0]/3), round(layer_5.shape[1]/3):-round(layer_5.shape[1]/3)]
    neg_layer_5 = layer_5[round(layer_5.shape[0]/3):-round(
        layer_5.shape[0]/3), round(layer_5.shape[1]/3):-round(layer_5.shape[1]/3)]
    try:
        pos_peak = np.unravel_index(
            pos_layer_5.argmax(), pos_layer_5.shape)
    except ValueError:
        raise

    peak = (pos_peak[0]+round(layer.shape[0]/3),
            pos_peak[1]+round(layer.shape[1]/3))

    return peak