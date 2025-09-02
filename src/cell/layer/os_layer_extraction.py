from typing import List, Dict
from collections import OrderedDict
from pathlib import Path
import json
import PIL
import svgpathtools
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import os
import math

from src.cell.layer.helpers import gaussian_filter_nan, get_cube_center

import PIL.Image, PIL.ImageDraw

DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), 'LayerExtractDebug')

def debug_visualize_mask(oct_file, seg_file, dims, subject_id=999, bscan_id=0, output_dir=DEFAULT_OUTPUT_DIR):
    # Load the OCT image and SVG segmentation paths
    oct_img = np.array(PIL.Image.open(oct_file))
    svg_paths = svgpathtools.svg2paths(str(seg_file))[0]
    
    # Here you would normally extract layer points and combine them.
    # For demonstration, we create an artificial polygon.
    # Uncomment and modify the following lines if you have real points.
    # points = np.array([complex(10,10), complex(200,10), complex(200,80), complex(10,80)])
    # polygon_coords = [(p.real, p.imag) for p in points]
    
    # Build the mask image.
    mask_img = PIL.Image.new("L", (dims['x'], dims['z']), 0)
    draw = PIL.ImageDraw.Draw(mask_img)
    # Example: draw.polygon(polygon_coords, outline=1, fill=1)
    
    # Convert to boolean array.
    mask_array = np.array(mask_img, dtype=bool)

    # Plot the raw B-scan and overlay the mask.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(oct_img, cmap='gray')
    ax1.set_title("Raw B-scan")
    
    ax2.imshow(oct_img, cmap='gray')
    ax2.imshow(mask_array, alpha=0.4, cmap='Reds')
    ax2.set_title("Mask overlaid")
    
    # Add a suptitle including subject and B-scan info.
    plt.suptitle(f"Subject {subject_id} - B-scan {bscan_id}")

    # Save the figure in a subject-specific subfolder.
    subject_folder = os.path.join(output_dir, str(subject_id))
    os.makedirs(subject_folder, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(subject_folder, f"debug_visualize_mask_subj_{subject_id}_bscan_{bscan_id}.png")
    fig.savefig(file_name)
    print("DEBUG:Mask Image saved to", file_name)
    plt.close(fig)



def debug_dark_peak_process(y, peaks, dark_peaks, dark_peak, _left, _right, p, subject_id=999, bscan_id=0, output_dir=DEFAULT_OUTPUT_DIR):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(y, label='Intensity')
    
    # Mark primary peaks.
    plt.plot(peaks, y[peaks], 'ro', label='Primary Peaks')
    
    # Invert the intensity between the first two peaks for visualization.
    inverted = 255 - y[peaks[0]:peaks[1]+1]
    inverted_x = np.arange(peaks[0], peaks[1]+1)
    plt.plot(inverted_x, 255 - inverted, 'c--', label='Inverted Region')
    
    # Mark detected dark peaks.
    plt.plot(dark_peaks, y[dark_peaks], 'kx', markersize=10, label='Dark Peaks')
    
    # Mark the chosen dark peak.
    plt.plot(dark_peak, y[dark_peak], 'ms', markersize=10, label='Chosen Dark Peak')
    
    # Plot the quadratic fit over the window.
    fit_x = np.arange(_left, _right)
    fit_y = np.polyval(p, fit_x)
    plt.plot(fit_x, fit_y, 'g-', label='Quadratic Fit')
    
    # Compute and mark the vertex of the quadratic fit.
    vertex = -p[1] / (2 * p[0])
    vertex_val = np.polyval(p, vertex)
    plt.plot(vertex, vertex_val, 'yo', markersize=10, label='Quadratic Vertex')
    
    plt.title(f"Dark Peak Debugging - Subject {subject_id}, B-scan {bscan_id}")
    plt.xlabel("Vertical index")
    plt.ylabel("Intensity")
    plt.legend()
    
    # Save the figure in a subject-specific subfolder.
    subject_folder = os.path.join(output_dir, str(subject_id))
    os.makedirs(subject_folder, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(subject_folder, f"debug_dark_peak_process_subj_{subject_id}_bscan_{bscan_id}.png")
    plt.savefig(file_name)
    print("DEBUG:Dark Peak Image saved to", file_name)
    plt.close(fig)



def debug_cropped_region(oct_img, mask, z_min, z_max, subject_id=999, bscan_id=0, output_dir=DEFAULT_OUTPUT_DIR):
    # Apply the mask to the OCT image.
    masked_oct = oct_img * mask

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(masked_oct, cmap='gray')
    ax1.set_title("Full masked OCT")

    cropped = masked_oct[z_min:z_max+1, :]
    ax2.imshow(cropped, cmap='gray')
    ax2.set_title("Cropped region")
    
    plt.suptitle(f"Subject {subject_id} - B-scan {bscan_id}")
    
    # Save the figure in a subject-specific subfolder.
    subject_folder = os.path.join(output_dir, str(subject_id))
    os.makedirs(subject_folder, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(subject_folder, f"debug_cropped_region_subj_{subject_id}_bscan_{bscan_id}.png")
    fig.savefig(file_name)
    print("DEBUG:Cropped Region Image saved to", file_name)
    plt.close(fig)

def debug_peaks_in_column(oct_column, column_index, subject_id=999, bscan_id=0, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Plot intensity vs. row index, then highlight the peaks found by scipy.signal.find_peaks.
    """

    fig = plt.figure(figsize=(6, 4))
    # plt.figure(figsize=(6, 4))
    plt.plot(oct_column, label='Intensity')
    
    # Find peaks.
    peaks_idx = scipy.signal.find_peaks(oct_column)[0]
    plt.scatter(peaks_idx, oct_column[peaks_idx], color='red', label='Peaks')
    
    plt.title(f"Column {column_index} intensity profile - Subject {subject_id}, B-scan {bscan_id}")
    plt.xlabel("Vertical index")
    plt.ylabel("Intensity")
    plt.legend()
    
    # Save the figure in a subject-specific subfolder.
    subject_folder = os.path.join(output_dir, str(subject_id))
    os.makedirs(subject_folder, exist_ok=True)
    file_name = os.path.join(subject_folder, f"debug_peaks_in_column_{column_index}_subj_{subject_id}_bscan_{bscan_id}.png")
    plt.savefig(file_name)

    print("DEBUG: Peaks Image saved to", file_name)
    fig.savefig(file_name)
    plt.close()


def debug_overlay_boundaries(oct_cropped, upper_points, lower_points, left, right, 
                             z_shift=0, subject_id=999, bscan_id=0, smoothing=False, 
                             output_dir=DEFAULT_OUTPUT_DIR, zoom_fraction=0.7):
    """
    Save the cropped OCT image with detected 'upper' and 'lower' boundaries overlaid,
    and zoom in on the region where the boundaries lie.
    
    Parameters:
        oct_cropped: 2D numpy array, the cropped OCT image.
        upper_points: array-like, row indices for the upper boundary at each column x.
        lower_points: array-like, row indices for the lower boundary at each column x.
        left: int, the left boundary of the horizontal range used.
        right: int, the right boundary of the horizontal range used.
        z_shift: int, how much was cut off from the top.
        subject_id: int or str, identifier for the subject.
        bscan_id: int, identifier for the B-scan.
        smoothing: bool, flag to indicate if boundaries are smoothed.
        output_dir: str, base directory where subject subfolders will be created.
        zoom_fraction: float, fraction (0-1) of the boundaries’ bounding box to display.
                      For example, 0.5 displays the central 50% of the boundary region.
    """
    # Create subject-specific folder and a subfolder (Raw/Smooth).
    subject_folder = os.path.join(output_dir, str(subject_id))
    os.makedirs(subject_folder, exist_ok=True)
    subfolder = "Smooth" if smoothing else "Raw"
    subfolder_path = os.path.join(subject_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    
    # Build the file name.
    file_name = f"subject_{subject_id}_bscan_{bscan_id}_debug_overlay"
    if smoothing:
        file_name += "_smoothed"
    file_name += ".png"
    filename = os.path.join(subfolder_path, file_name)
    print("DEBUG: Saving image to", filename)
    
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(oct_cropped, cmap='gray', origin='upper')
    
    # Compute x-values for the boundaries.
    x_vals = np.arange(left, right + 1)
    
    # Adjust the boundaries using z_shift.
    adjusted_upper = np.array(upper_points) 
    adjusted_lower = np.array(lower_points) 
    
    # Plot the boundary lines.
    ax.plot(x_vals, adjusted_upper, 'g-', label='Upper boundary')
    ax.plot(x_vals, adjusted_lower, 'r-', label='Lower boundary')
    ax.set_aspect('equal')
    ax.set_title(f"Detected boundaries in cropped region for subject {subject_id}")
    # ax.legend()
    
    # Compute a bounding box that covers the boundaries.
    x_min = left
    x_max = right
    y_min = min(np.min(adjusted_upper), np.min(adjusted_lower))
    y_max = max(np.max(adjusted_upper), np.max(adjusted_lower))
    
    # Determine the center of the boundaries.
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Determine the full width and height of the boundaries.
    full_width = (x_max - x_min)
    full_height = (y_max - y_min)
    
    # Define the zoom window as a fraction of the boundaries' bounding box.
    window_width = full_width * zoom_fraction
    window_height = full_height * zoom_fraction
    
    # Set the x limits based on the computed center and window size.
    ax.set_xlim(center_x - window_width / 2 + 50, center_x + window_width / 2 - 50)
    # For the y-axis, because origin='upper', the limits are reversed.
    ax.set_ylim(center_y + window_height / 2 + 2.5, center_y - window_height / 2 - 2.5)
    
    plt.savefig(filename)
    plt.close(fig)
    print(f"Boundaries Image saved to: {filename}")

def debug_plot_thickness(thicknesses, upper_boundary, lower_boundary, bscan_id, subject_id, spacing, output_dir=DEFAULT_OUTPUT_DIR):
    
    """
    Plot the thicknesses of the OS layer extracted from a B-scan.
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot thickness on the primary y-axis
    ax.plot(thicknesses/spacing, label='Thickness (microm)', color='blue')
    ax.set_xlabel("Horizontal index")
    ax.set_ylabel("Thickness (microm)")

    ax.set_aspect('equal')

    # Create a secondary y-axis for boundaries
    ax2 = ax.twinx()
    ax2.plot(upper_boundary, label='Upper Boundary', color='green')
    ax2.plot(lower_boundary, label='Lower Boundary', color='red')
    ax2.invert_yaxis()  # Flip the y-axis for boundaries
    ax2.set_ylabel("Boundary Value")

    ax2.set_aspect('equal')
    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_title(f"OS layer thicknesses for subject {subject_id}, B-scan {bscan_id}")


    # Save the figure.
    os.makedirs(output_dir, exist_ok=True)
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #file_name = os.path.join(output_dir, f"debug_plot_thickness_{timestamp}.png")

    file_name = os.path.join(output_dir, str(subject_id),  f"debug_plot_thickness_{subject_id}_{bscan_id}.png")
    print("DEBUG: Thickness Image saved to", file_name)
    fig.savefig(file_name)
    plt.close(fig)



from typing import List, Dict
from collections import OrderedDict
from pathlib import Path
import json
import PIL
import PIL.ImageDraw
import svgpathtools
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt



from src.cell.layer.helpers import gaussian_filter_nan, get_cube_center

def get_OS_thicknesses(subject_id: int, dataset_path: Path, radius: float = 1, debug: bool = False) -> Dict:
    """
    Get the thicknesses of the OS layer for the given subject, using the segmented
    layers of the OCT cube in the provided dataset_path. The thicknesses are calculated
    in a square of provided radius (in mm) around the center of the cube, which
    is defined as the location of the bright white dot.
        
    ### This function is meant to be used by `save_layer_features.ipynb`, saving the result
    to corresponding `Subject*/Session*/layer_new/OS_thicknesses.json`.
    
    :param subject_id: The subject id.
    :type subject_id: int
    :param dataset_path: The path to the OCT cube dataset.
    :type dataset_path: Path
    :param radius: The radius around the center of the cube in mm.
    :type radius: float
    :param debug: If True, display debug visualizations.
    :type debug: bool
    :return: The thicknesses of the OS layer in mm.
    :rtype: OrderedDict
    """

    with open(r'V:\Studies\AOSLO\data\cohorts\AOSLO healthy\lut_subject_to_white_dot_bscan.json', 'r') as f:
        lut_subject_to_bscan = json.load(f)
        strSubjId = str(subject_id)
    if strSubjId not in lut_subject_to_bscan:
        raise ValueError()
    try:
        bscan_center = lut_subject_to_bscan[strSubjId] - 1
    except:
        print(f"Error: {strSubjId} not found in lut_subject_to_bscan")
        raise ValueError()

    dir_oct = dataset_path / 'oct' / 'volume'
    with open(dir_oct / 'info.json') as json_file:
        infos = json.load(json_file)
    dims = {'x': infos['shape'][3], 'y': infos['shape'][0], 'z': infos['shape'][2]}
    spacing = {'x': infos['spacing'][3], 'y': infos['spacing'][0], 'z': infos['spacing'][2]}
    center_y, center_x = get_cube_center(subject_id)
    radius = 1  # in mm, around the center (overwrites input radius)
    radius_x = int(radius / spacing['x'])
    radius_y = int(radius / spacing['y'])

    result = OrderedDict()
    for bscan in range(center_y - radius_y, center_y + radius_y + 1):
        bscan_id = f'{bscan:04d}'
        seg_file = dataset_path / 'children' / 'segmentation' / f'{bscan_id}.svg'
        oct_file = dir_oct / f'{bscan_id}.jpg'

        # Read segmentation (SVG) and OCT image
        svg_paths: List[svgpathtools.Path] = svgpathtools.svg2paths(str(seg_file))[0]
        oct = np.array(PIL.Image.open(oct_file))
        # print(bscan-bscan_center)
        if debug and int(bscan) == int(bscan_center):
            # Debug: visualize the raw segmentation mask 
            debug_visualize_mask(oct_file, seg_file, dims, subject_id=subject_id, bscan_id=bscan)

        # Define helper lambdas to filter and extract points from SVG paths
        check_point = lambda point: 0.025 < point.real / dims['x'] < 0.975 and 0 < point.imag / dims['z'] < 1
        extract_layer = lambda i: np.array([line.start for line in svg_paths[i] if check_point(line.start)])
        points_ONL = extract_layer(4)
        points_PRRPE = extract_layer(5)
        points_CC = extract_layer(6)

        # Extract the upper and lower interfaces of the RP_RPE layer
        upper_int = np.intersect1d(points_ONL, points_PRRPE)
        lower_int = np.intersect1d(points_PRRPE, points_CC)

        # Reorder the points in trigonometric order
        upper_int = upper_int[np.argsort(upper_int.real)[::-1]]
        lower_int = lower_int[np.argsort(lower_int.real)]
        points = np.r_[upper_int, lower_int]
        
        # Create a mask using the polygon defined by the interfaces
        mask_img = PIL.Image.new("L", (dims['x'], dims['z']), 0)
        PIL.ImageDraw.Draw(mask_img).polygon(list(zip(points.real, points.imag)), outline=1, fill=1)
        mask = np.array(mask_img, dtype=bool)
        
        #if debug and int(bscan) == int(bscan_center):
        #     # directly visualize the mask
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     ax.imshow(oct, cmap='gray', origin='upper')
        #     ax.imshow(mask, alpha=0.4, cmap='Reds')
        #     ax.set_title(f"B-scan {bscan_center} mask overlay")
        #     plt.show()

        # Determine cropping indices based on the mask
        z_nonzero = mask.nonzero()[0]
        z_shift = z_nonzero.min()
        z_max = z_nonzero.max()
        if debug and int(bscan) == int(bscan_center):
            #Debug: visualize the full masked OCT and the cropped region
            debug_cropped_region(oct, mask, z_shift, z_max, bscan_id = bscan, subject_id = subject_id)
        
        oct_pr = (oct * mask)[z_shift:z_max+1, :]
        height, width = oct_pr.shape[:2]

        threshold = np.quantile(oct_pr[oct_pr > 0], 0.3)

        upper_boundary = []
        lower_boundary = []
        for i in range(width // 4, 3 * width // 4):
            shift = np.where(oct_pr[:, i] > 0)[0].min()
            y = gaussian_filter_nan(oct_pr[oct_pr[:, i] > 0, i].astype(float), 1)
            peaks = scipy.signal.find_peaks(y)[0]
            if len(peaks) < 2:
                continue
            dark_peaks = peaks[0] + scipy.signal.find_peaks(255 - y[peaks[0]:peaks[1] + 1])[0]
            if len(dark_peaks) == 0:
                continue
            dark_peak = dark_peaks[np.argmin(y[dark_peaks])]

            # Quadratic fit around the dark peak
            _left = max(peaks[0], dark_peak - 2)
            _right = min(peaks[1], dark_peak + 2) + 1
            p = np.polyfit(np.arange(_left, _right), y[_left:_right], 2)
            v_peak = np.polyval(p, -p[1] / (2 * p[0]))
            if debug and i == width // 2 and int(bscan) == int(bscan_center):  # For example, for the first column or a particular column of interest
                debug_dark_peak_process(y, peaks, dark_peaks, dark_peak, _left, _right, p, subject_id=subject_id, bscan_id=bscan)
                # debug_plot_column_location(oct_pr, i)


            # Adjust threshold based on peak value
            factor = 1 if v_peak >= threshold else 1 - 0.2 * (1 - v_peak / threshold)
            under_threshold = peaks[0] + np.where(y[peaks[0]:peaks[1] + 1] < factor * threshold)[0]
            if under_threshold.size == 0:
                continue

            # Record the boundary positions (i, vertical index)
            upper_boundary.append((i, shift + under_threshold.min()))
            lower_boundary.append((i, shift + under_threshold.max()))

            if debug and i == width // 2 and int(bscan) == int(bscan_center):
                # Debug: visualize the intensity profile for the middle column
                debug_peaks_in_column(y, i, bscan_id=bscan, subject_id=subject_id)

        left = max(min(upper_boundary + lower_boundary, key=lambda x: x[0])[0], center_x - radius_x)
        right = min(max(upper_boundary + lower_boundary, key=lambda x: x[0])[0], center_x + radius_x)
        x = np.arange(left, right + 1)
        upper_interp = np.interp(x, *zip(*upper_boundary))
        lower_interp = np.interp(x, *zip(*lower_boundary))

        # if debug: #and int(bscan) == int(bscan_center):
        #     # Debug: overlay the interpolated upper and lower boundaries on the cropped region
        #     debug_overlay_boundaries(oct_pr, upper_interp, lower_interp, left, right, z_shift,
        #                             subject_id, bscan_id=bscan, smoothing=False,
        #                             )


        window_length = 11  # must be an odd integer, adjust based on your data resolution
        polyorder = 2

        upper_smoothed = scipy.signal.savgol_filter(upper_interp, window_length, polyorder)
        lower_smoothed = scipy.signal.savgol_filter(lower_interp, window_length +10, polyorder)


        if debug: #and int(bscan) == int(bscan_center):
            # Debug: overlay the interpolated upper and lower boundaries on the cropped region
            debug_overlay_boundaries(oct_pr, upper_smoothed, lower_smoothed, left, right, z_shift, 
                                    subject_id, bscan_id=bscan, smoothing=True,
                                    )

        thicknesses = (lower_smoothed - upper_smoothed) * spacing['z']

        if debug and int(bscan) == int(bscan_center):
            print(thicknesses)
            debug_plot_thickness(thicknesses, upper_boundary = upper_smoothed, lower_boundary = lower_smoothed, bscan_id = bscan, subject_id = subject_id, spacing = spacing['z'])


        result.setdefault(bscan_id, {})
        result[bscan_id]['vector'] = [
            None if t == np.inf else t
            for t in np.pad(thicknesses, (left, dims['x'] - right - 1), 'constant', constant_values=np.inf).tolist()
        ]
        result[bscan_id]['center'] = center_x
        result[bscan_id]['average'] = np.nanmean(thicknesses)
        print ("done with bscan", bscan_id)
    
    return {
        dataset_path.name: result,
        'spacing': spacing,
    }
