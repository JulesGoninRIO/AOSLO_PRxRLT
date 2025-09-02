from typing import Tuple
from skimage.metrics import structural_similarity
from sklearn.metrics import normalized_mutual_info_score

import numpy as np
from src.Helpers.Exceptions import ZeroVector, EmptyVector

def normalized_cross_correlation(a: np.array, b: np.array) -> float:
    """
    Computes the Normalized Cross Correlation between two vectors

    :param a: the first vector
    :type a: np.array
    :param b: the second vector
    :type b: np.array
    :raise AssertionError: if the length of the vectors are not the same
    :raises EmptyVector: if one of the vector is empty
    :raises ZeroVector: if the vector contains only zeros
    :return: the NCC between the vectors
    :rtype: float
    """
    assert len(a)==len(b), f"The two vectors must be the same length"
    if len(a) == 0:
        raise EmptyVector(f"{a} is empty")
    if len(b) == 0:
        raise EmptyVector(f"{b} is empty")
    if np.std(a) == 0:
        raise ZeroVector(f"{a} standard deviation is zero")
    if np.std(b) == 0:
        raise ZeroVector(f"{b} standard deviation is zero")
    std_a = np.std(a)
    std_b = np.std(b)
    a_n = a / std_a
    b_n = b / std_b
    return (1/len(a))*(np.correlate(a_n.flatten(), b_n.flatten())[0])

def zero_normalized_cross_correlation(a: np.array, b: np.array) -> float:
    """
    Computes the Zero Normalized Cross Correlation between two vectors

    :param a: the first vector
    :type a: np.array
    :param b: the second vector
    :type b: np.array
    :raise AssertionError: if the length of the vectors are not the same
    :raises EmptyVector: if one of the vector is empty
    :raises ZeroVector: if the vector contains only zeros
    :return: the ZNCC between the vectors
    :rtype: float
    """
    assert len(a)==len(b), f"The two vectors must be the same length"
    if len(a) == 0:
        raise EmptyVector(f"{a} is empty")
    if len(b) == 0:
        raise EmptyVector(f"{b} is empty")
    if np.std(a) == 0:
        raise ZeroVector(f"{a} standard deviation is zero")
    if np.std(b) == 0:
        raise ZeroVector(f"{b} standard deviation is zero")
    a_zn = (a - np.mean(a)) / np.std(a)
    b_zn = (b - np.mean(b)) / np.std(b)
    return (1/len(a))*(np.correlate(a_zn.flatten(), b_zn.flatten())[0])

def mutual_information(hgram: np.ndarray) -> float:
    """
    Mutual Information (MI) for joint histogram
    https://matthew-brett.github.io/teaching/mutual_information.html

    :param hgram: histogram of pixel values
    :return: mutual information score
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def normalized_mean_square_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Normalized Mean Square Error of two arrays

    :param a: the first vector
    :type a: np.ndarray
    :param b: the second vector
    :type b: np.ndarray
    :raise AssertionError: if the length of the vectors are not the same
    :raises EmptyVector: if one of the vector is empty
    :return: the result of the NMSE
    :rtype: float
    """

    assert len(a)==len(b), f"The two vectors must be the same length"
    if len(a) == 0:
        raise EmptyVector(f"{a} is empty")
    if len(b) == 0:
        raise EmptyVector(f"{b} is empty")

    return ((a.flatten() - b.flatten())**2).mean(axis=None)

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Structural Similarity Index Measure of two arrays

    :param a: the first vector
    :type a: np.ndarray
    :param b: the second vector
    :type b: np.ndarray
    :raise AssertionError: if the length of the vectors are not the same
    :raises EmptyVector: if one of the vector is empty
    :return: the SSIM score
    :rtype: float
    """

    assert a.shape==b.shape, f"The two vectors must be the same length"
    if len(a) == 0:
        raise EmptyVector(f"{a} is empty")
    if len(b) == 0:
        raise EmptyVector(f"{b} is empty")
    return structural_similarity(a.flatten(), b.flatten())

def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Normalized Mutual Information score of two arrays

    :param a: the first vector
    :type a: np.ndarray
    :param b: the second vector
    :type b: np.ndarray
    :raise AssertionError: if the length of the vectors are not the same
    :raises EmptyVector: if one of the vector is empty
    :return: the NMI score
    :rtype: float
    """

    assert a.shape==b.shape, f"The two vectors must be the same length"
    if len(a) == 0:
        raise EmptyVector(f"{a} is empty")
    if len(b) == 0:
        raise EmptyVector(f"{b} is empty")
    return normalized_mutual_info_score(a.flatten(), b.flatten())

def entropy(img_hist: np.ndarray) -> float:
    """
    Comptues the entropy of an array

    :param img_hist: array containing image histogram values
    :type img_hist: np.ndarray
    :return: entropy of the array
    :rtype: float
    """

    img_hist = img_hist / float(np.sum(img_hist))
    img_hist = img_hist[np.nonzero(img_hist)]

    return -np.sum(img_hist * np.log2(img_hist))
