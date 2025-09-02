import numpy as np
import cv2
import pandas as pd
import re
import cv2
import numpy as np
from sklearn.utils import resample
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage import transform as tf
import os
from sklearn.utils import resample
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage import transform as tf

from src.shared.numpy.arrays import rotate_image
from src.cell.montage.ssim import SSIM

def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """
    Return True if the center element of the 3x3x3 input array is strictly
    greater than or less than all its neighbors, False otherwise
    """
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_subimage) and \
                   np.all(center_pixel_value >= third_subimage) and \
                   np.all(center_pixel_value >= second_subimage[0, :]) and \
                   np.all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_subimage) and \
                   np.all(center_pixel_value <= third_subimage) and \
                   np.all(center_pixel_value <= second_subimage[0, :]) and \
                   np.all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                               second_image[i-1:i+2, j-1:j+2],
                               third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.all(np.abs(extremum_update) < 0.5):
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if np.abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))
            keypoint.response = np.abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    return 0.5 * (pixel_array[[1, 1, 2], [1, 2, 1], [2, 1, 1]] - pixel_array[[1, 1, 0], [1, 0, 1], [0, 1, 1]])

def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.1, scale_factor=1.0):
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = np.max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = np.bitwise_and(keypoint.octave, 255)
    layer = np.bitwise_and(np.right_shift(keypoint.octave, 8), 255)
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def initialize_bounding_box(NumOfRefs):
    maxXRef = np.full(NumOfRefs, -1e10)
    minXRef = np.full(NumOfRefs, 1e10)
    maxYRef = np.full(NumOfRefs, -1e10)
    minYRef = np.full(NumOfRefs, 1e10)
    Xwidth = np.zeros(NumOfRefs)
    Ywidth = np.zeros(NumOfRefs)
    return maxXRef, minXRef, maxYRef, minYRef, Xwidth, Ywidth

def pad_image(im, vr, ur):
    nonzero = im[:,:, 1] > 0
    im = im[:,:,0]
    if im.shape[0] < len(vr):
        im = np.pad(im, ((0, len(vr) - im.shape[0]), (0, 0)))
        nonzero = np.pad(nonzero, ((0, len(vr) - nonzero.shape[0]), (0, 0)))
    if im.shape[1] < len(ur):
        im = np.pad(im, ((0, 0), (0, len(ur) - im.shape[1])))
        nonzero = np.pad(nonzero, ((0, 0), (0, len(ur) - nonzero.shape[1])))

    return im, nonzero

def get_best_transform(X1, X2, n, matchTolerance, rotLimit):
    scores = []
    all_Hs = []
    bestScore = -1
    bestH = np.eye(3)
    bestOK_all = np.zeros(X1.shape[1], dtype=bool)
    bestScale = 1
    numMatches_all = X1.shape[1]
    allIndex = np.arange(numMatches_all)  # list of all the indexes for the matches
    # maybe need to augment because we don't want similar points, base = 5000
    for _ in range(n):
        # estimate model
        if numMatches_all >= 3:
            subset = resample(allIndex, n_samples=3, replace=False)
        else:
            subset = allIndex

        H = np.eye(3)
        TransRadians = 0
        Scale = 1
        # Score Using Only rotation + translation
        # Compute the Procrustes transformation)
        try:
            _, _, tform = procrustes(X2[:2, subset].T, X1[:2, subset].T,
                                     scaling=False, reflection=False)
        except ValueError:
            continue
        # Construct the homography matrix
        H = np.vstack([np.column_stack([tform['rotation'],
                                        tform['translation']]), [0, 0, 1]])

        # Compute the rotation angle in radians
        TransRadians = np.arctan2(H[1, 0], H[0, 0])
        X2_ = H @ X1
        du = X2_[0, :] / X2_[2, :] - X2[0, :] / X2[2, :]
        dv = X2_[1, :] / X2_[2, :] - X2[1, :] / X2[2, :]
        #TODO: sometimes warning:
        # C:\Users\BardetJ\Downloads\aoslo_pipeline-master\src\PostProc_Pipe\
        # Montaging\AOAutomontagingPython\sift_mosaic_fast_MultiModal.py:224:
        # RuntimeWarning: overflow encountered in multiply
        # ok = du * du + dv * dv < matchTolerance * matchTolerance
        ok = du * du + dv * dv < matchTolerance * matchTolerance
        score = np.sum(ok)
        scores.append(score)
        all_Hs.append(H)

        if score > bestScore and abs(TransRadians) < rotLimit and Scale < 1.1 and Scale > 0.9:
            bestScore = score
            bestH = H
            bestOK_all = ok
            bestScale = Scale

    return bestH, scores, all_Hs, bestOK_all, bestScale, numMatches_all

def procrustes(X, Y, scaling=True, reflection='best'):
    # From https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    if normX == 0 or normY == 0:
        raise ValueError('Procrustes error: X or Y have no variation')

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def sift_mosaic_fast_MultiModal(im1, im2, f1, d1, f2, d2, rotLimit):
    rotLimit = np.pi / 18 if rotLimit is None else rotLimit
    matchTolerance = 6
    total_f1, total_f2 = sum([len(f) for f in f1]), sum([len(f) for f in f2])
    if total_f1 < 1 or total_f2 < 1:
        return np.eye(3), 0, 0, 1
    number_of_iterations = 50000

    X1, X2, matches, numMatches = initialize_arrays(number_of_iterations, len(f1))
    X1, X2, matches, numMatches = match_features(f1, d1, f2, d2, X1, X2, matches, numMatches)
    X1, X2 = add_homogeneous_coordinate(X1, X2)

    bestH, scores, all_Hs, bestOK_all, bestScale, numMatches_all = get_best_transform(X1, X2, number_of_iterations, matchTolerance, rotLimit)
    # numOkMatches, bestOK = get_num_ok_matches(numMatches, bestOK_all)

    base = True
    if not base:
        try:
            bestH = process_images(im1, im2, bestH, scores, all_Hs)
        except:
            pass

    return bestH, np.sum(bestOK_all), numMatches_all, bestScale, scores

def initialize_arrays(number_of_iterations, MN):
    X1 = np.empty((2, number_of_iterations), dtype=np.float32)
    X2 = np.empty((2, number_of_iterations), dtype=np.float32)
    matches = np.empty(MN, dtype=object)
    numMatches = np.zeros(MN, dtype=int)
    return X1, X2, matches, numMatches

def match_features(f1, d1, f2, d2, X1, X2, matches, numMatches):
    for m in range(len(f1)):
        try:
            bf = cv2.BFMatcher()
            matches_m = bf.knnMatch(np.array(d1[m]),np.array(d2[m]),k=2)
        except cv2.error:
            matches_m = []

        good = [i for i, j in matches_m if i.distance < 0.8 * j.distance]
        X1_m, X2_m, matches_m = remove_duplicates(f1, f2, m, good)

        matches[m] = matches_m
        if len(X1_m)==0:
            numMatches[m] = 0
        else:
            numMatches[m] = X1_m.shape[1]
        X1[:, :numMatches[m]] = X1_m
        X2[:, :numMatches[m]] = X2_m

    return X1, X2, matches, numMatches

def remove_duplicates(f1, f2, m, good):
    X1_m = np.float32([f1[m][match.queryIdx].pt for match in good]).T
    X2_m = np.float32([f2[m][match.trainIdx].pt for match in good]).T
    _, IA = np.unique(np.round(np.vstack([X1_m, X2_m])).T, axis=0, return_index=True)
    if len(X1_m)==0 and len(X2_m)==0:
        return X1_m, X2_m, []
    elif len(X1_m)==0:
        X2_m = X2_m[:, IA]
    elif len(X2_m)==0:
        X1_m = X1_m[:, IA]
    else:
        X1_m, X2_m = X1_m[:, IA], X2_m[:, IA]
    matches_m = [good[i] for i in IA]
    return X1_m, X2_m, matches_m

def add_homogeneous_coordinate(X1, X2):
    X1 = np.concatenate((X1, np.ones((1, X1.shape[1]))), axis=0)
    X2 = np.concatenate((X2, np.ones((1, X2.shape[1]))), axis=0)
    return X1, X2

# def get_num_ok_matches(numMatches, bestOK_all):
#     numOkMatches = np.zeros(len(numMatches))
#     bestOK = [None] * len(numMatches)
#     offset = 0
#     for m in range(len(numMatches)):
#         bestOK[m] = bestOK_all[offset:offset+numMatches[m]]
#         numOkMatches[m] = np.sum(bestOK[m])
#         offset += numMatches[m]
#     return numOkMatches, bestOK

def process_images(im1, im2, bestH, scores, all_Hs):

    try:
        im1_8bit = convert_to_8bit(im1[0])
        im2_8bit = convert_to_8bit(im2[0])

        angle = np.arctan2(bestH[0, 1], bestH[0, 0])
        rotated_image = rotate_image(im2_8bit, np.rad2deg(angle))
        diff_shape = [rotated_image.shape[0]-im1_8bit.shape[0],
                    rotated_image.shape[1]-im1_8bit.shape[1]]

        pad1y, pad1x = calculate_padding(bestH, diff_shape)
        pad2y, pad2x = calculate_padding(bestH, diff_shape)

        try:
            im1_8bit = pad_image(im1_8bit, pad1y, pad1x)
            rotated_image = pad_image(rotated_image, pad2y, pad2x)
        except ValueError:
            print("too big")
        # BGR
        final_image = im1_8bit + rotated_image
        # cv2.imwrite(os.path.join(r"C:\Users\BardetJ\Downloads", f"yesay.tif"), final_image.astype(np.uint8))
        im1_bool = np.array(im1_8bit, dtype=bool)
        rotated_image_bool = np.array(rotated_image, dtype=bool)

        # Create a mask where both images are present
        mask = np.logical_and(im1_bool, rotated_image_bool)
        ssim_class = SSIM(im1_8bit, 'balec', rotated_image, 'balec2', mask, r'C:\Users\BardetJ\Downloads', 10)
        ssim_class.run()
        ssim_score = ssim_class.m
        # increase size of figure
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(ssim_score, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(os.path.join(r'C:\Users\BardetJ\Downloads', "ssim_heatmap.png"))
        # plt.close()
        # print("oe")

        # Sort the scores in descending order
        sorted_scores = sorted(scores, reverse=True)

        # Initialize the result list with the first score
        result = [sorted_scores[0]]

        # Iterate over the rest of the scores
        for score in sorted_scores[1:]:
            # If the score is the same as the last one in the result list, continue to the next score
            if score == result[-1]:
                continue
            # Otherwise, add the score to the result list
            result.append(score)
            # If we have 5 scores in the result list, break the loop
            if len(result) == 10:
                break

        # Get all indices of the values in result from the original scores list
        indices = [[i for i, x in enumerate(scores) if x == value] for value in result]

        # Initialize the result list
        result_indices = []

        # Iterate over the list of lists
        for sublist in indices:
            # Check if adding the current sublist would result in more than 5 indices in the result
            if len(result_indices) + len(sublist) > 10:
                break
            # Otherwise, add the indices of the sublist to the result list
            result_indices.extend(sublist)

        # Trim the result list to only include the first 5 indices
        result_indices = result_indices[:10]

        max_scores = []
        Hs = []
        # TODO: WARNING: result indices are sometimes empty
        for i, largest_arg in enumerate(result_indices):
            best_transform = all_Hs[largest_arg]
            Hs.append(best_transform)

            image1_8bit = convert_to_8bit(im1[0])
            image2_8bit = convert_to_8bit(im2[0])

            angle = np.arctan2(best_transform[0, 1], best_transform[0, 0])
            rotated_image = rotate_image(image2_8bit, np.rad2deg(angle))

            diff_shape = [rotated_image.shape[0] - image1_8bit.shape[0],
                        rotated_image.shape[1] - image1_8bit.shape[1]]

            pad_y, pad_x = calculate_padding(best_transform, diff_shape)

            image1_8bit = pad_image(image1_8bit, pad_y, pad_x)
            rotated_image = pad_image(rotated_image, pad_y, pad_x)

            # BGR
            final_image = im1_8bit + rotated_image
            cv2.imwrite(os.path.join(r"C:\Users\BardetJ\Downloads", f"yesay_{i}.tif"), final_image.astype(np.uint8))
            im1_bool = np.array(im1_8bit, dtype=bool)
            rotated_image_bool = np.array(rotated_image, dtype=bool)

            # Create a mask where both images are present
            mask = np.logical_and(im1_bool, rotated_image_bool)
            ssim_class = SSIM(im1_8bit, 'balec', rotated_image, 'balec2', mask, r'C:\Users\BardetJ\Downloads', 10)
            ssim_class.run()
            ssim_score = ssim_class.m
            max_scores.append(np.max(ssim_score))
            # increase size of figure
            # fig = plt.figure(figsize=(10, 10))
            # plt.imshow(ssim_score, cmap='hot', interpolation='nearest')
            # plt.colorbar()
            # plt.tight_layout()
            # plt.savefig(os.path.join(r'C:\Users\BardetJ\Downloads', f"ssim_heatmap_{i}.png"))
            # plt.close()

        bestH = Hs[np.argmax(max_scores)]
    except:
        pass

    return bestH

def convert_to_8bit(image):
    return (image[0] * 255).astype(np.uint8)

def pad_image(image, pad_y, pad_x):
    try:
        return np.pad(image, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])))
    except ValueError:
        print("too big")
        return image

def calculate_padding(best_transform, diff_shape):
    pad_x = [0, 0]
    pad_y = [0, 0]

    if best_transform[1, 2] < 0:
        pad_y = [0, round(abs(best_transform[1, 2])) + diff_shape[0]]
    else:
        pad_y = [round(abs(best_transform[1, 2])), diff_shape[0]]

    if best_transform[0, 2] < 0:
        pad_x = [0, round(abs(best_transform[0, 2])) + diff_shape[1]]
    else:
        pad_x = [round(abs(best_transform[0, 2])), diff_shape[1]]

    return pad_y, pad_x
