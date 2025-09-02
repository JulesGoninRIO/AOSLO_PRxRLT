from typing import Dict, Callable, Tuple, List
import json  
from pathlib import Path
from PIL import Image, ImageDraw
import svgpathtools
import numpy as np
import pandas as pd
import scipy.optimize

from src.cell.layer.helpers import gaussian_filter_nan

###
# THE FOVEAL SHAPE EXTRACTION STEP IS MANAGED BY src/save_layer_features.ipynb
###

def get_upper_lower_seg(svg: List[svgpathtools.Path], dims: Dict[str,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the upper and lower segmentation lines from the SVG paths and returns the indices and coordinates of the lines.

    :param svg: The SVG paths to extract the segmentation from.
    :type svg: List[svgpathtools.Path]
    :param dims: The dimensions of the cube.
    :type dims: Dict[str,int]
    :return: The indices and coordinates of the upper and lower lines.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    EPS = 0.025
    check_point = lambda point: EPS < point.real / dims['x'] < 1 - EPS and \
                                EPS < point.imag / dims['z'] < 1 - EPS
    points_tot = np.array([
        line.start
        for line in svg[0] # 1st path is the total retina (all layers together)
        if check_point(line.start)
    ])
    points_cc = np.array([
        line.start
        for line in svg[6] # 7th path is the choroid layer (for the baseline)
        if check_point(line.start)
    ])
    # lower choroid layer limit (== lower all_layers limit)
    lower_cc = np.intersect1d(points_tot, points_cc)
    # upper all_layers limit
    upper = np.setdiff1d(points_tot, lower_cc)
    # upper choroid layer limit (= baseline)
    lower = np.setdiff1d(points_cc, lower_cc)
    # trim off the first and last points to avoid issues with segmentation
    left_lim = int(np.ceil(max(lower.real.min(), upper.real.min())))
    right_lim = 1 + int(np.floor(min(lower.real.max(), upper.real.max())))
    # reorder the points to have them in trigonometric order
    upper = upper[np.argsort(upper.real)[::-1]]
    lower = lower[np.argsort(lower.real)]
    points = np.r_[upper, lower]
    # create a mask to extract the upper and lower lines
    mask = Image.new("L", (dims['x'], dims['z']), 0)
    ImageDraw.Draw(mask).polygon(list(zip(points.real, points.imag)), outline=255, fill=255)
    mask = np.array(mask, dtype=bool)
    indices = np.arange(left_lim, right_lim)
    upper_line = mask.shape[0] - mask[:, indices].argmax(axis=0)
    lower_line = mask[:, indices][::-1].argmax(axis=0)
    return indices, upper_line, lower_line

def process_segmentation(dir_to_process: Path | str, spacing: Dict[str,float], dims: Dict[str,int]) -> np.ndarray:
    """
    Extracts the segmentation from the SVG files in the given directory and returns the 3D coordinates of the segmentation, using the given spacing (having structure {'x': float, 'y': float, 'z': float}).
    
    :param dir_to_process: The directory containing the SVG files.
    :type dir_to_process: Path | str
    :param spacing: The spacing between the points in the x, y, and z directions.
    :type spacing: Dict[str,float]
    :return: The 3D coordinates of the segmentation.
    :rtype: np.ndarray
    """
    if isinstance(dir_to_process, str):
        dir_to_process = Path(dir_to_process)
        
    X, Y = np.meshgrid(spacing['x'] * np.arange(dims['x']), spacing['y'] * np.arange(dims['y']), indexing='ij')
    Z_up = np.nan * np.ones((dims['x'], dims['y']))
    Z_lo = np.nan * np.ones((dims['x'], dims['y']))
    for svg_file in dir_to_process.glob('*.svg'):
        svg, _ = svgpathtools.svg2paths(str(svg_file))
        if len(svg) < 3:
            continue
        try:
            indices, upper_line, lower_line = get_upper_lower_seg(svg, dims)
        except Exception as e:
            # print(repr(e))
            continue
        i = int(svg_file.stem)
        Z_up[indices, i] = spacing['z'] * upper_line
        Z_lo[indices, i] = spacing['z'] * lower_line
    return X, Y, Z_up, Z_lo

def fit_curve_l1(x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 model: Callable, 
                 p0: np.ndarray,
                 lower_bound: np.ndarray | List | None = None,
                 upper_bound: np.ndarray | List | None = None,
                 f_scale: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    if lower_bound is None:
        lower_bound = np.zeros_like(p0)
    if upper_bound is None:
        upper_bound = np.inf
    residuals_fun = lambda p, x, y, z: model(x, y, p) - z
    results = scipy.optimize.least_squares(residuals_fun, p0, args=(x, y, z), loss='soft_l1', f_scale=f_scale, bounds=(lower_bound, upper_bound))
    return results.x, results.fun

def fit_poly_gauss(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cx: float, cy: float) -> Tuple[np.ndarray, Callable] | None:
    """
    Fits a polynomial and a 2D Gaussian to the given data and returns the parameters of the fits. Returns None if the data is not sufficient. The model reads
    Z = A00 + A10 * X + A01 * Y + A20 * X^2 + A02 * Y^2 + A11 * X * Y - depth * exp(-((X - center_x)^2 / (2 * width_x^2)) - ((Y - center_y)^2 / (2 * width_y^2)))
    
    :param X: The x-coordinates of the data points.
    :type X: np.ndarray
    :param Y: The y-coordinates of the data points.
    :type Y: np.ndarray
    :param Z: The z-coordinates of the data points.
    :type Z: np.ndarray
    :param cx: The approximate x-coordinate of the center of the fovea.
    :type cx: float
    :param cy: The approximate y-coordinate of the center of the fovea.
    :type cy: float
    :return: The parameters of the fits.
    :rtype: Tuple[List[float]] | None
    """
    # keep data points in central disk
    central_disk = np.sqrt((X - cx)**2 + (Y - cy)**2) < 2.2
    X_cd = X[central_disk]
    Y_cd = Y[central_disk]
    Z_cd = Z[central_disk]
    lower_bd  = [-1, -10, -10,  0,  0, -10, 0.01, cx-1, 0.1, cy-1, 0.1]
    p0        = [ 0,   0,   0, .5, .5,   0,  0.1,   cx, 0.5,   cy, 0.5]
    higher_bd = [ 1,  10,  10,  1,  1,  10,    1, cx+1, 0.9, cy+1, 0.9]
    model_poly_gauss = lambda x, y, p: p[0] + p[1] * x + p[2] * y - p[3] * x**2 - p[4] * y**2 + p[5] * x * y - p[6] * np.exp(-((x - p[7])**2 / (2 * p[8]**2)) - ((y - p[9])**2 / (2 * p[10]**2)))
    popt_pg, _ = fit_curve_l1(X_cd, Y_cd, Z_cd, model_poly_gauss, p0, lower_bd, higher_bd)
    
    return popt_pg, lambda x, y: model_poly_gauss(x, y, popt_pg)

def get_max_slope(p: np.ndarray) -> float:
    df_dx = lambda x, y: p[1] - 2 * p[3] * x + p[5] * y + p[6] * (x - p[7]) / p[8]**2 * np.exp(-(x - p[7])**2 / (2 * p[8]**2) - ((y - p[9])**2 / (2 * p[10]**2)))
    df_dy = lambda x, y: p[2] - 2 * p[4] * y + p[5] * x + p[6] * (y - p[9]) / p[10]**2 * np.exp(-(x - p[7])**2 / (2 * p[8]**2) - ((y - p[9])**2 / (2 * p[10]**2)))
    constraint = lambda x: 7 - (x[0] - p[7])**2 / (2 * p[8]**2) - (x[1] - p[9])**2 / (2 * p[10]**2) # >= 0
    constraint_jac = lambda x: np.array([-(x[0] - p[7]) / p[8]**2, -(x[1] - p[9]) / p[10]**2])
    objective = lambda x: -np.linalg.norm([df_dx(x[0], x[1]), df_dy(x[0], x[1])])
    result = scipy.optimize.minimize(objective, [3, 3], constraints={'type': 'ineq', 'fun': constraint, 'jac': constraint_jac}, method='SLSQP')
    return -result.fun

def get_flatness(popt: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    c_x, w_x, c_y, w_y = popt[7:]
    Z_smoothed = gaussian_filter_nan(Z, 2)
    Z_smoothed += gaussian_filter_nan(Z - Z_smoothed, 2)
    ell_mask = (X - c_x)**2 / w_x**2 + (Y - c_y)**2 / w_y**2 <= 0.6**2
    dz_dx, dz_dy = np.gradient(Z_smoothed, X[:,0], Y[0,:])
    gradient_norm = np.sqrt(dz_dx**2 + dz_dy**2)
    flatness = 1 / np.quantile(gradient_norm[ell_mask], 0.4)
    return flatness

def get_volume(popt: np.ndarray) -> float:
    return 2 * np.pi * popt[6] * popt[8] * popt[10]

def extract_foveal_shape_params(data: Tuple[np.ndarray],
                                range_: Dict[str,float], 
                                data_to_csv: str | None = None) -> np.ndarray | None:
    """
    Extracts the shape parameters of the foveal pit from the given data. Return None if the data is not sufficient, otherwise return the fitted parameters as well as maximum slope.

    :param data: The data to extract the shape parameters from, shape (N,3)
    :type data: np.ndarray
    :param range_: The range of the data in the x, y, and z directions.
    :type range_: Dict[str,int]
    :return: The fitted parameters and maximum slope.
    :rtype: np.ndarray | None
    """
    assert all(d.shape == data[0].shape for d in data), 'Data have not the same shape'
    if len(data) == 4:
        X, Y, Z_up, Z_lo = data
        Z = Z_up - Z_lo
    elif len(data) == 3:
        X, Y, Z = data
    else:
        print('Data should have 3 or 4 elements')
        return None
    
    # first approximation of where the foveal center should be
    center_x = range_['x'] / 2
    center_y = range_['y'] / 2

    # downsample to speed up fitting, points far from (3,3) are less likely to be selected
    TARGET = 30000 # target number of points to fit the model
    if (size := np.sum(~np.isnan(Z))) < 1000:
        print(f'Not enough data points, only {size}')
        return None
    elif size > TARGET:
        # trust the logic
        shift = 1.58853363 - np.log(1.03617677 * size / TARGET - 1) / 1.62750008
        probabilities = np.exp(-(np.maximum(np.sqrt((X - center_x)**2 + (Y - center_y)**2) - shift, 0) / 0.9)**2)
        mask = np.random.uniform(0, 1, size=X.shape) < probabilities
        flat_X = X[mask].ravel()
        flat_Y = Y[mask].ravel()
        flat_Z = Z[mask].ravel()
    else:
        flat_X = X.ravel()
        flat_Y = Y.ravel()
        flat_Z = Z.ravel()
    notna = ~np.isnan(flat_Z)

    # fit the model
    result = fit_poly_gauss(flat_X[notna], flat_Y[notna], flat_Z[notna], center_x, center_y)

    if result is None:
        return None
    popt, model = result

    if data_to_csv is not None:
        pd.DataFrame({'X': flat_X[notna], 'Y': flat_Y[notna], 'Z': flat_Z[notna]}).to_csv(data_to_csv, index=False, float_format='%.5g')

    max_slope = get_max_slope(popt)
    flatness = get_flatness(popt, X, Y, Z)
    volume = get_volume(popt)

    # recall parameters: 'A00', 'A10', 'A01', 'A20', 'A02', 'A11', 'depth', 'center_x', 'width_x', 'center_y', 'width_y', 'max_slope', 'flatness'
    indices = ['A00', 'A10', 'A01', 'A20', 'A02', 'A11', 'depth', 'center_X', 'width_X', 'center_Y', 'width_Y', 'max_slope', 'flatness', 'volume']
    # return dataframe containing popts in columns, using indices for rows
    df = pd.DataFrame({'params': np.r_[popt, max_slope, flatness, volume]}, index=indices)
    return df

def get_oct_shape_3d(dataset_path: Path | str, name: str) -> Tuple[np.ndarray, Dict[str,float]] | None:
    """
    Process the segmentation in the given study and extract the shape parameters of the foveal pit.
    
    :param dataset_path: The study to process.
    :type dataset_path: Path | str
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    svg_dir = dataset_path / 'children' / 'segmentation'
    if not svg_dir.exists():
        print(f'{name}: No segmentation directory found in {dataset_path}')
        return None
    
    info_path = dataset_path / 'oct' / 'volume' / 'info.json'
    if not info_path.exists():
        print(f'{name}: No spacing file found in {dataset_path}')
        return None
    with open(info_path) as info_file:
        infos = json.load(info_file)

    spacing = {'x': infos['spacing'][3], 'y': infos['spacing'][0], 'z': infos['spacing'][2]}
    dims = {'x': infos['shape'][3], 'y': infos['shape'][0], 'z': infos['shape'][2]}
    _range = {'x': infos['range'][3], 'y': infos['range'][0], 'z': infos['range'][2]}

    assert dims['y'] > 1, f'{name}: Not enough B-scans in the volume'
    
    return process_segmentation(svg_dir, spacing, dims), _range