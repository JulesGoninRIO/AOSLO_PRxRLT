import cv2
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from joblib import Parallel, delayed
# from .helpers import filter_sift_features_by_roi
from .SIFT import SIFT
import multiprocessing
import time
# from memory_profiler import memory_usage
import psutil
import os

def measure_performance(func, *args, **kwargs):
    pass
    # # Measure CPU
    # p = psutil.Process(os.getpid())
    # start_cpu = p.cpu_times()

    # # Measure time
    # start_time = time.time()
    # result = func(*args, **kwargs)
    # end_time = time.time()
    # elapsed_time = end_time - start_time

    # # Measure memory
    # mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=1)

    # # Measure CPU
    # end_cpu = p.cpu_times()
    # cpu_usage_user = end_cpu.user - start_cpu.user
    # cpu_usage_system = end_cpu.system - start_cpu.system

    # return result, elapsed_time, max(mem_usage), cpu_usage_user, cpu_usage_system

def calculateFeatures(imageFilename: list, pixelScale: list, MN: int, N: int):
    """
    Calculate features for each image.

    :param imageFilename: The filenames of the images
    :type imageFilename: list
    :param pixelScale: The pixel scale for each image
    :type pixelScale: list
    :param MN: The number of modalities
    :type MN: int
    :param N: The number of images
    :type N: int
    :return: The features and descriptors for each image
    :rtype: tuple
    """
    # Stores features for each image
    f_all = [[None]*N for _ in range(MN)]
    d_all = [[None]*N for _ in range(MN)]

    def list_to_keypoints(list):
        return [cv2.KeyPoint(x[0], x[1], x[2], _angle=x[3], _response=x[4],
                             _octave=x[5], _class_id=x[6]) for x in list]

    def process_n(n, MN, imageFilename, pixelScale):
        pixelScale_n = pixelScale[n]
        results = Parallel(n_jobs=-1)(delayed(process_image)(m,
                                                             n,
                                                             imageFilename,
                                                             pixelScale_n) for m in range(MN))
        return results

    print("Running SIFT in parallel")
    num_cores = multiprocessing.cpu_count()
    while num_cores > 0:
        try:
            results = Parallel(n_jobs=num_cores)(delayed(process_n)(n, MN, imageFilename, pixelScale) for n in range(N))
            break  # If successful, break out of the loop
        except np.core._exceptions._ArrayMemoryError:
            print(f"num_core: {num_cores}")
            num_cores -= 1  # Decrement the number of cores and try again
            print("restarting with num_cores: ", num_cores)

    if num_cores == 0:
        raise RuntimeError("Failed to process with any number of cores")
    for n in range(len(results)):
        for m in range(MN):
            f_all[m][n] = list_to_keypoints(results[n][m][0])
            d_all[m][n] = results[n][m][1]
    # for n in range(N):
    #     for m in range(MN):
    #         if imageFilename[m][n]:  # If this file is blank, then that means we don't have valid information for it- skip it.
    #             f1_crop, d1_crop = process_image(m, n, imageFilename, pixelScale[n])
    #             f_all[m][n] = f1_crop
    #             d_all[m][n] = d1_crop

    return f_all, d_all

def process_image(m: int, n: int, imageFilename: list, pixelScale_n: float):
    """
    Process a single image to extract features.

    :param m: The modality index
    :type m: int
    :param n: The image index
    :type n: int
    :param imageFilename: The filenames of the images
    :type imageFilename: list
    :param pixelScale_n: The pixel scale for the image
    :type pixelScale_n: float
    :return: The keypoints and descriptors for the image
    :rtype: tuple
    """
    im = cv2.imread(imageFilename[m][n], cv2.IMREAD_GRAYSCALE)
    new_size = (int(im.shape[0]*pixelScale_n), int(im.shape[1]*pixelScale_n))
    resized_im = resize(im, new_size, mode='reflect')
    im = img_as_ubyte(resized_im)

    sift = SIFT(im)
    # # Usage:
    # result, elapsed_time, max_mem_usage, cpu_usage_user, cpu_usage_system = measure_performance(sift.computeKeypointsAndDescriptors)
    # print(f"Elapsed time: {elapsed_time} seconds")
    # print(f"Max memory usage: {max_mem_usage} MiB")
    # print(f"CPU usage: {cpu_usage_user} percent, system: {cpu_usage_system}")
    # print(result)
    result = sift.computeKeypointsAndDescriptors()
    f1 = result[0]
    d1 = result[1]
    # we have to remove the Keypoints object to serialize the results when running
    # in parallel
    norm_image = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_image = cv2.convertScaleAbs(norm_image)
    # img_with_keypoints = cv2.drawKeypoints(norm_image, f1_crop,
    # None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite(r"C:\Users\BardetJ\Downloads\keypoints.tif", img_with_keypoints)
    def keypoints_to_list(keypoints):
        return [(kp.pt[0], kp.pt[1], kp.size, kp.angle,
                 kp.response, kp.octave, kp.class_id) for kp in keypoints]
    return keypoints_to_list(f1), d1