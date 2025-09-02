import cv2
import numpy as np
from sklearn.utils import resample
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage import transform as tf
import os
# from scipy.spatial import procrustes
from sklearn.utils import resample
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage import transform as tf
from src.PostProc_Pipe.Helpers.arrays import rotate_image
from src.PostProc_Pipe.Montaging.SSIM import SSIM

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

    X1, X2, matches, numMatches = initialize_arrays(total_f1, total_f2, len(f1))
    X1, X2, matches, numMatches = match_features(f1, d1, f2, d2, X1, X2, matches, numMatches)
    X1, X2 = add_homogeneous_coordinate(X1, X2)

    bestH, scores, all_Hs, bestOK_all, bestScale, numMatches_all = get_best_transform(X1, X2, 50000, matchTolerance, rotLimit)
    numOkMatches, bestOK = get_num_ok_matches(numMatches, bestOK_all)

    base = True
    if not base:
        try:
            bestH = process_images(im1, im2, bestH, scores, all_Hs)
        except:
            pass

    return bestH, scores, all_Hs, bestOK_all, bestScale, numMatches_all, numOkMatches, bestOK

def initialize_arrays(total_f1, total_f2, MN):
    X1 = np.empty((2, total_f1), dtype=np.float32)
    X2 = np.empty((2, total_f2), dtype=np.float32)
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
        numMatches[m] = X1_m.shape[1]
        X1[:, :numMatches[m]] = X1_m
        X2[:, :numMatches[m]] = X2_m

    return X1, X2, matches, numMatches

def remove_duplicates(f1, f2, m, good):
    X1_m = np.float32([f1[m][match.queryIdx].pt for match in good]).T
    X2_m = np.float32([f2[m][match.trainIdx].pt for match in good]).T
    _, IA = np.unique(np.round(np.vstack([X1_m, X2_m])).T, axis=0, return_index=True)
    X1_m, X2_m = X1_m[:, IA], X2_m[:, IA]
    matches_m = [good[i] for i in IA]
    return X1_m, X2_m, matches_m

def add_homogeneous_coordinate(X1, X2):
    X1 = np.concatenate((X1, np.ones((1, X1.shape[1]))), axis=0)
    X2 = np.concatenate((X2, np.ones((1, X2.shape[1]))), axis=0)
    return X1, X2

def get_num_ok_matches(numMatches, bestOK_all):
    numOkMatches = np.zeros(len(numMatches))
    bestOK = [None] * len(numMatches)
    offset = 0
    for m in range(len(numMatches)):
        bestOK[m] = bestOK_all[offset:offset+numMatches[m]]
        numOkMatches[m] = np.sum(bestOK[m])
        offset += numMatches[m]
    return numOkMatches, bestOK

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
