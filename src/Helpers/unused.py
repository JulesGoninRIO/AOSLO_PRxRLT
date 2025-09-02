# import os
# import shutil

# import numpy as np
# from numba import njit
# import glob
# import csv
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# import cv2
# import pandas as pd
# from tqdm import tqdm
# import seaborn as sns
# from sklearn.mixture import GaussianMixture
# from scipy.stats import norm
# import math
# from pathlib import Path
# import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.spatial import KDTree
# from scipy.spatial import cKDTree
# import timeit
# from typing import List, Dict, Tuple, Callable, Set
# import logging
# import pickle
# # from Configs.Parser import Parser
# import tifffile
# from PIL import Image

# from .thresholding import separate_dark_regions

# import matplotlib
# matplotlib.use('Agg')

# def compare_montage_to_transform(shift_matlab, montage_image, image_dir):
#     """ TO COMPLETE TO SEE THE DIFFERENCE BETWEEN THE OUTPUT IMAGE AND MATLAB'S TRANSFORM"""
#     tot_image = np.zeros((11720, 9604))
#     for shift_image, shift_value in shift_matlab.items():
#         image = cv2.imread(os.path.join(image_dir, shift_image), cv2.IMREAD_GRAYSCALE)
#         # image = np.pad(image, ((4049, 4049),(4442, 4442)), 'constant')
#         image = np.pad(image, ((5500, 5500),(4442, 4442)), 'constant')
#         transform = np.array(eval(shift_value))[:2,:]
#         transform[:,2:3] = -transform[:,2:3]
#         image = cv2.warpAffine(image, transform, (image.data.shape[1], image.data.shape[0]))
#         # limits = update_limits(image, limits)
#         tot_image += image

#     filled_pixels = np.argwhere(tot_image > 0)
#     top = np.min(filled_pixels[:, 0])
#     bottom = np.max(filled_pixels[:, 0])
#     left = np.min(filled_pixels[:, 1])
#     right = np.max(filled_pixels[:, 1])
#     image = image[top:bottom+1, left:right+1]

#     cv2.imwrite(r"C:\Users\BardetJ\Documents\figures_alain\montage.png", tot_image.astype(np.uint8))

#     import pdb
#     pdb.set_trace()
#     return 0

# def get_neighbors(montage_dir: str):
#     # ATTENTION GET NEIGHBORS LOOK FOR SUBFOLDERS -> USE SUBJECT NIEGHBOR
#     # SHOULD IMPROVE IT
#     """
#     Get all the neighbor images from a patient

#     :param montage_dir: path where the folder with the montage images are
#     :return neighbors_image:
#     """
#     neighbors_image = {}
#     shift_dirs = []
#     for folder in os.listdir(montage_dir) :
#         subject_dir = os.path.join(montage_dir, folder)
#         if os.path.isdir(subject_dir):
#             for session in os.listdir(subject_dir) :
#                 session_dir = os.path.join(subject_dir, session)
#                 montaged_dir = os.path.join(session_dir, "corrected")
#                 shift_dirs.append(montaged_dir)

#     # shift_dirs = [r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1\Session455\corrected",
#     #                 r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1\Session456\corrected",
#     #                 r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1\Session457\corrected",
#     #                 r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1\Session458\corrected",
#     #                 r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1\Session459\corrected",
#     #                 r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1\Session460\corrected",
#     #                 r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_first_try\Subject1\Session461\corrected"]

#     for shift_dir in shift_dirs :
#         # import pdb
#         # pdb.set_trace()
#         image_neighbors = subject_neighbor(shift_dir)

#     return neighbors_image

# def prepare_folders(base_dir: str) -> list:
#     """ Prepare the data for montaging by looking into the structure of the data """
#     input_dir = [base_dir]
#     register_dirs = []
#     processing = True
#     print(f"Processing data for {input_dir}")
#     while processing :
#         filenames = os.listdir(input_dir[0])
#         if all([os.path.isdir(os.path.join(input_dir[0], filename)) for filename in filenames]):
#             # We have only subfolders so let's dive into it
#             input_dir.extend([os.path.join(input_dir[0], filename) for filename in filenames])
#             input_dir.pop(0)
#         else :
#             if not "CalculatedSplit" in input_dir[0] :
#                 register_dirs.append(input_dir[0])
#             input_dir.pop(0)
#             if not input_dir:
#                 print("Finished preprocessing data for montage")
#                 processing = False
#         #elif any([os.path.join(input_dir[0], filename).endswith(".tif") for filename in filenames]):
#     return register_dirs

# def neighbors(locations: List[Tuple[int, int]]) -> dict:
#     """ Find neighbors in list of tuples """ #TODO
#     neighbors = {}
#     for location in locations :
#         #import pdb
#         #pdb.set_trace()
#         neighbor_to = location
#         distance = np.inf
#         neighbors[location] = []
#         for other_location in locations:
#             if other_location != location :
#                 import pdb
#                 pdb.set_trace()
#                 dist_to_coords = np.abs(np.array((other_location[0] - location[0], other_location[1] - location[1])))
#                 sum_distance = dist_to_coords[0] - location[0] + dist_to_coords[1] - location[1]
#                 if sum_distance < distance :
#                     distance = sum_distance
#                     neighbors[location].append(other_location)
#                 if sum_distance == distance :
#                     distance = sum_distance
#                     neighbors[location].append(other_location)
#     return neighbors

# def analyze_mikhail_nst():
#     """TO MODIFY -> LOOK AT GET DISTRIBUTION IN ANANLYZER"""
#     """ Compare results of numebers of cones detected between Mikhail's algorithm and Davidson's with NST"""
#     # montage_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_"
#     # std_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_done\std_both_border_global"
#     # output_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_done\var_both_border_global"

#     # # MIKHAIL_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_mikhail_toy\postprocessed"
#     # # # DAVIDSON_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_wrong\first_run\output_cone_detector\algorithmMarkers"
#     # # DAVIDSON_DIR_first_run = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_done\first_run"
#     # # DAVIDSON_DIR_nst_bad = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_done\var_both_border_global_nst"
#     # # DAVIDSON_DIR_nst_both = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_done\var_both_border_global_nst_both"
#     # # OUTPUT_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\mikhail_comparison"

#     # montage_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_real_"
#     # std_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\std_both_border_global"
#     # output_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\var_both_border_global"

#     # MIKHAIL_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_mikhail_real\postprocessed"
#     # # DAVIDSON_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_wrong\first_run\output_cone_detector\algorithmMarkers"
#     # DAVIDSON_DIR_first_run = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\first_run"
#     # # DAVIDSON_DIR_nst_bad = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\var_both_border_global_nst"
#     # DAVIDSON_DIR_nst_both = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\var_both_border_nst_both"
#     # OUTPUT_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\mikhail_comparison"

#     # montage_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\w_dark_alain"
#     # std_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_alain\std_both_border_global"
#     # output_dir = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\var_both_border_global"

#     # # MIKHAIL_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_mikhail_real\postprocessed"
#     # # # DAVIDSON_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline_wrong\first_run\output_cone_detector\algorithmMarkers"
#     # # DAVIDSON_DIR_first_run = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\first_run"
#     # # # DAVIDSON_DIR_nst_bad = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\var_both_border_global_nst"
#     # # DAVIDSON_DIR_nst_both = r"C:\Users\BardetJ\Documents\mikhail\dataset\pipeline\var_both_border_nst_both"
#     # # OUTPUT_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\mikhail_comparison"

#     MIKHAIL_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\few_images_for_alain\cs\postprocessed"
#     DAVIDSON_DIR_first_run = r"C:\Users\BardetJ\Documents\mikhail\dataset\few_images_for_alain\cs\var_both_border_global"
#     DAVIDSON_DIR_nst_both = r"C:\Users\BardetJ\Documents\mikhail\dataset\few_images_for_alain\cs\var_both_border_nst"
#     OUTPUT_DIR = r"C:\Users\BardetJ\Documents\mikhail\dataset\figures_alain"

#     # number_of_cones_per_images_first_run = get_cones(DAVIDSON_DIR_first_run)
#     # number_of_cones_per_images_nst_both = get_cones(DAVIDSON_DIR_nst_both)

#     # for key, val in number_of_cones_per_images_first_run.items():
#     #     count_first_run = len(np.where(val)[0])

#     # for key, val in number_of_cones_per_images_nst_both.items():
#     #     count_nst_both = len(np.where(val)[0])

#     # print(count_first_run, count_nst_both)

#     densities = {}

#     number_of_cones_per_images_first_run = get_cones(DAVIDSON_DIR_first_run)
#     number_of_cones_per_images_nst_both = get_cones(DAVIDSON_DIR_nst_both)

#     count_first_run = []
#     count_nst_both = []

#     for key, val in number_of_cones_per_images_first_run.items():
#         count_first_run.append(len(np.where(val)[0]))

#     for key, val in number_of_cones_per_images_nst_both.items():
#         count_nst_both.append(len(np.where(val)[0]))

#     print(np.mean(count_first_run), np.mean(count_nst_both))
#     print(np.std(count_first_run), np.std(count_nst_both))

#     count_same  = 0
#     for i in range(len(count_first_run)):
#         if count_first_run[i] == count_nst_both[i] :
#             count_same+=1
#     print(count_same, len(count_first_run))

#     # bad_images = ["_8120_", "_8115_", "_8142_", "_8150_", "_8152_", "_8143_", "_8170_", "_8171_", "_8172_", "_8173_", "_8174_", "_8169_", "_8168_", "_8169_", "_8166_", "_8167_"]
#     bad_images = []

#     mikhail_eccentricities = np.linspace(0,3,4)
#     davidson_eccentricities = np.append([0], np.linspace(3,10,8))

#     # mikhail_count = {"1": [], "2": []}
#     mikhail_count = {"0": [], "1": [], "2": [], "3": []}
#     davidson_count_first_run = {"3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
#     davidson_count_nst_bad = {"3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
#     davidson_count_nst_both = {"3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
#     davidson_count_nst_std = {"3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}

#     used_images = {}
#     for csv_file in os.listdir(MIKHAIL_DIR):
#         if csv_file.endswith(".csv") :
#             image_name = re.sub("CROP_(\d+)_x(\d+)y(\d+)_", "", csv_file)
#             location = re.search("\((.*?)\)", image_name).group(1)
#             image_number = re.search("_(\d+)_", image_name).group(0)
#             if image_number in bad_images:
#                 print(f"{image_name} is so baaaaaaaaaaaaaaaaaad")
#                 continue
#             location = eval(location)
#             if np.abs(location[0]) in mikhail_eccentricities and np.abs(location[1]) in mikhail_eccentricities \
#                 and np.abs(location[0])+np.abs(location[1]) < 4:
#                 count = 0
#                 with open(os.path.join(MIKHAIL_DIR, csv_file), "r") as f :
#                     reader = csv.reader(f)
#                     for row in reader:
#                         count+=1
#                 if image_name not in used_images.keys() :
#                     used_images[image_name] = count
#                 else :
#                     used_images[image_name] += count

#     for image, count in used_images.items():
#         location = re.search("\((.*?)\)", image).group(1)
#         location = eval(location)
#         relative_location = np.abs(location[0]) + np.abs(location[1])
#         image_number = re.search("_(\d+)_", image).group(0)
#         mikhail_count[str(relative_location)].append(count)
#         cell_density = count/(IMAGE_SIZE*IMAGE_SIZE)
#         densities[image] = cell_density

#     cones_tot_location_image_first_run = get_cones(DAVIDSON_DIR_first_run)
#     area_dark_regions = {}
#     dark_regions_to_discard = {}
#     for image_number, _ in cones_tot_location_image_first_run.items():
#         image_name = [image_ for image_ in os.listdir(Path(MIKHAIL_DIR).parent) if image_number in image_]
#         image_name = [image_ for image_ in image_name if "Confocal" in image_][0]
#         if not image_name :
#             continue
#         image = cv2.imread(os.path.join(Path(MIKHAIL_DIR).parent, image_name), cv2.IMREAD_GRAYSCALE)
#         dark_regions = separate_dark_regions(image, image_name)
#         dark_regions_to_discard[image_number] = dark_regions
#         area_dark_regions[image_number] = 1 - (np.count_nonzero(dark_regions)/(dark_regions.shape[0]*dark_regions.shape[1]))

#     for image_number, val in cones_tot_location_image_first_run.items():
#         count = 0
#         for cone in np.argwhere(val):
#             if dark_regions_to_discard[image_number][cone[0], cone[1]] :
#                 count+=1
#         # count = len(np.where(val)[0])
#         # if count < 1000 :
#             # continue
#         if image_number in bad_images :
#             continue
#         image_name = [image_ for image_ in os.listdir(Path(MIKHAIL_DIR).parent) if image_number in image_]
#         if not image_name :
#             continue
#         location = re.search("\((.*?)\)", image_name[0]).group(1)
#         location = eval(location)
#         if np.abs(location[0]) in davidson_eccentricities and np.abs(location[1]) in davidson_eccentricities :
#             relative_location = np.abs(location[0]) + np.abs(location[1])
#             if relative_location == 0: continue
#             davidson_count_first_run[str(relative_location)].append(count/(1-area_dark_regions[image_number]))

#     # cones_tot_location_image_nst_bad = get_cones(DAVIDSON_DIR_nst_bad)
#     # for image_number, val in cones_tot_location_image_nst_bad.items():
#     #     count = 0
#     #     for cone in np.argwhere(val):
#     #         if dark_regions_to_discard[image_number][cone[0], cone[1]] :
#     #             count+=1
#     #     if count < 1000 :
#     #         continue
#     #     if image_number in bad_images :
#     #         continue
#     #     image_name = [image_ for image_ in os.listdir(Path(MIKHAIL_DIR).parent) if image_number in image_]
#     #     if not image_name :
#     #         continue
#     #     location = re.search("\((.*?)\)", image_name[0]).group(1)
#     #     location = eval(location)
#     #     if np.abs(location[0]) in davidson_eccentricities and np.abs(location[1]) in davidson_eccentricities :
#     #         relative_location = np.abs(location[0]) + np.abs(location[1])
#     #         davidson_count_nst_bad[str(relative_location)].append(count/(1-area_dark_regions[image_number]))

#     cones_tot_location_image_nst_both = get_cones(DAVIDSON_DIR_nst_both)
#     for image_number, val in cones_tot_location_image_nst_both.items():
#         count = 0
#         for cone in np.argwhere(val):
#             if dark_regions_to_discard[image_number][cone[0], cone[1]] :
#                 count+=1
#         # if count < 1000 :
#         #     continue
#         if image_number in bad_images :
#             continue
#         image_name = [image_ for image_ in os.listdir(Path(MIKHAIL_DIR).parent) if image_number in image_]
#         if not image_name :
#             continue
#         location = re.search("\((.*?)\)", image_name[0]).group(1)
#         location = eval(location)
#         if np.abs(location[0]) in davidson_eccentricities and np.abs(location[1]) in davidson_eccentricities \
#             and np.abs(location[0])+np.abs(location[1])>2 :
#             relative_location = np.abs(location[0]) + np.abs(location[1])
#             if relative_location == 0: continue
#             count = count/(1-area_dark_regions[image_number])
#             davidson_count_nst_both[str(relative_location)].append(count)
#             cell_density = count/(IMAGE_SIZE*IMAGE_SIZE)
#             if image_name[0] in densities.keys():
#                 densities[image_name[0]] = (densities[image_name]+cell_density)/2
#             else :
#                 densities[image_name[0]] = cell_density

#     # # import pdb
#     # # pdb.set_trace()
#     # cones_detected = cones_per_patch(output_dir)
#     # number_of_cones_per_patch = list(cones_detected.values())
#     # max_of_cones_per_patch = np.max(number_of_cones_per_patch)
#     # # The histogram of cones detected per patch
#     # cones_detected_hist = np.histogram(number_of_cones_per_patch, max_of_cones_per_patch, [0,max_of_cones_per_patch])
#     # threshold = get_otsu_threshold(cones_detected_hist)
#     # patch_dir = os.path.join(output_dir,"_raw")
#     # separated_patches = {"good": [], "bad": []}
#     # for patch in os.listdir(patch_dir) :
#     #     if cones_detected[patch] > threshold :
#     #         separated_patches["good"].append(patch)
#     #     else :
#     #         separated_patches["bad"].append(patch)

#     # corr_patch, incurrable_patches = get_corr_patch(output_dir, montage_dir, separated_patches, cones_detected, treat = "both", visualize = "False", solo = "True")
#     # # cones_location_patch = cones_location_per_patch(DAVIDSON_DIR_nst_both, std_dir, incurrable_patches)
#     # # cones_location_image = cones_location_per_image(cones_location_patch)
#     # # cones_tot_location_image_nst_std = get_overlaping_cones(cones_location_image)
#     # # cones_tot_location_image_nst_std = analyze_nst(output_dir, DAVIDSON_DIR_nst_both, corr_patch)
#     # cones_tot_location_image_nst_std = get_cones(DAVIDSON_DIR_nst_both, montage_dir, std_dir, incurrable_patches)

#     # # analyze_nst(first_dir: str, nst_dir: str, corr_patch:dict, filename_cones:dict = None)

#     # for image_number, val in cones_tot_location_image_nst_std.items():
#     #     count = 0
#     #     for cone in np.argwhere(val):
#     #         if dark_regions_to_discard[image_number][cone[0], cone[1]] :
#     #             count+=1
#     #     if count < 1000 :
#     #         continue
#     #     if image_number in bad_images :
#     #         continue
#     #     image_name = [image_ for image_ in os.listdir(Path(MIKHAIL_DIR).parent) if image_number in image_]
#     #     if not image_name :
#     #         continue
#     #     location = re.search("\((.*?)\)", image_name[0]).group(1)
#     #     location = eval(location)
#     #     if np.abs(location[0]) in davidson_eccentricities and np.abs(location[1]) in davidson_eccentricities :
#     #         relative_location = np.abs(location[0]) + np.abs(location[1])
#     #         if relative_location == 0: continue
#     #         davidson_count_nst_std[str(relative_location)].append(count/(1-area_dark_regions[image_number]))

#     # for csv_file in os.listdir(DAVIDSON_DIR):
#     #     image_name = re.sub("CROP_(\d+)_x(\d+)y(\d+)_", "", csv_file)
#     #     location = re.search("\((.*?)\)", image_name).group(1)
#     #     location = eval(location)
#     #     if np.abs(location[0]) in davidson_eccentricities and np.abs(location[1]) in davidson_eccentricities :
#     #         relative_location = np.abs(location[0]) + np.abs(location[1])
#     #         count = 0
#     #         with open(os.path.join(DAVIDSON_DIR, csv_file), "r") as f:
#     #             reader = csv.reader(f)
#     #             for row in reader:
#     #                 count+=1
#     #             davidson_count[str(relative_location)].append(count)

#     # x_mikhail = []
#     # y_mikhail = []
#     # y_mikhail_err = []
#     # x_davidson_first_run = []
#     # y_davidson_first_run = []
#     # y_davidson_first_run_err = []
#     # x_davidson_nst_bad = []
#     # y_davidson_nst_bad = []
#     # y_davidson_nst_bad_err = []
#     # x_davidson_nst_both = []
#     # y_davidson_nst_both = []
#     # y_davidson_nst_both_err = []
#     # x_davidson_nst_std = []
#     # y_davidson_nst_std = []
#     # y_davidson_nst_std_err = []

#     # # y_err_mikhail = []
#     # for key, val in mikhail_count.items():
#     #     x_mikhail.append(eval(key))
#     #     y_mikhail.append(np.sum(val)/len(val))
#     #     # y_mikhail.append(np.median(val))
#     #     y_mikhail_err.append(np.sqrt((np.sum(np.square(val-np.mean(val))))/len(val)))
#     # for key, val in davidson_count_first_run.items():
#     #     x_davidson_first_run.append(eval(key))
#     #     y_davidson_first_run.append(np.sum(val)/len(val))
#     #     # y_davidson_first_run.append(np.median(val))
#     #     y_davidson_first_run_err.append(np.sqrt((np.sum(np.square(val-np.mean(val))))/len(val)))
#     # # for key, val in davidson_count_nst_bad.items():
#     # #     x_davidson_nst_bad.append(eval(key))
#     # #     y_davidson_nst_bad.append(np.sum(val)/len(val))
#     # #     # y_davidson_nst_bad.append(np.median(val))
#     # #     y_davidson_nst_bad_err.append(np.sqrt((np.sum(np.square(val-np.mean(val))))/len(val)))
#     # for key, val in davidson_count_nst_both.items():
#     #     x_davidson_nst_both.append(eval(key))
#     #     y_davidson_nst_both.append(np.sum(val)/len(val))
#     #     # y_davidson_nst_both.append(np.median(val))
#     #     y_davidson_nst_both_err.append(np.sqrt((np.sum(np.square(val-np.mean(val))))/len(val)))
#     # # for key, val in davidson_count_nst_std.items():
#     # #     x_davidson_nst_std.append(eval(key))
#     # #     y_davidson_nst_std.append(np.sum(val)/len(val))
#     # #     # y_davidson_nst_std.append(np.median(val))
#     # #     y_davidson_nst_std_err.append(np.sqrt((np.sum(np.square(val-np.mean(val))))/len(val))) 

#     # plt.close("all")
#     # plt.errorbar(x = x_mikhail, y = y_mikhail, yerr = y_mikhail_err, color = "red", label="ATMS algorithm")
#     # plt.errorbar(x = x_davidson_first_run, y = y_davidson_first_run, yerr = y_davidson_first_run_err, color = "blue", label="Davidson's algorithm first run")
#     # # plt.errorbar(x = x_davidson_nst_bad, y = y_davidson_nst_bad, yerr = y_davidson_nst_bad_err, color = "purple", label="Davidson's NST algorithm on non-working patches")
#     # plt.errorbar(x = x_davidson_nst_both, y = y_davidson_nst_both, yerr = y_davidson_nst_both_err, color = "pink", label="Davidson's NST algorithm on both patches")
#     # # plt.errorbar(x = x_davidson_nst_std, y = y_davidson_nst_std, yerr = y_davidson_nst_std_err, color = "black", label="Davidson's NST algorithm on both patches and STD when NST not possible")
#     # plt.hlines(y=[5893, 3537, 2122], xmin=[0.85, 2.55, 4.25], xmax=[2.55, 4.25, 5.95], colors='aqua', linestyles='-', lw=2, label='From literature')
#     # plt.vlines(x=[1.7, 3.4, 5.2], ymin=[5893-602, 3537-31, 2122-275], ymax=[5893+602, 3537+31, 2122+275], colors='aqua', linestyles='-', lw=2)
#     # plt.title("Comparison of number of cones detected per location", fontsize=20)
#     # plt.xlabel("Relative eye location in degrees")
#     # plt.ylabel("Number of cones detected")
#     # plt.legend(fontsize = 12)
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(OUTPUT_DIR, "analysis_no_bad.png"))
#     # plt.close("all")

#     return densities

# #Regroup images from a session with their corresponding location that has the form (0,0)
# def regroup_location(session):
#     """Returns dictionary with locations and filenames corresponding to locations"""
#     locations={}
#     for image_name in session:
#         location = re.search('\((.*?)\)', image_name).group(1)
#         if location in locations.keys():
#             locations[location].append(image_name)
#         else:
#             locations[location] = [image_name]
#     return locations

# #Regroup session filenames by finding the pattern Session000 with 000 being the session number
# def regroup_session(IMAGES_DIR, endswith=None):
#     """Returns a dictionary with the session number as key and the list of all filenames corresponding to the session as value"""

#     sessions = {}
#     for image_name in sorted(os.listdir(IMAGES_DIR)):
#         if endswith is not None:
#             if not image_name.endswith(endswith):
#                 continue
#         #find the session number
#         session = re.search('Session(\d+)', image_name).group(1)
#         if session in sessions.keys():
#             sessions[session].append(image_name)
#         else:
#             sessions[session] = [image_name]
#     return sessions

# def separate_images_gmm(cones_detected: dict, gmm: GaussianMixture) -> dict:
#     """
#     Separate images from gaussian distributions

#     :param cones_detected: the number of cones detected per patch
#     :param gmm: the gaussian mixture model of the dataset
#     :return separated_images: the images separated based on their number of
#                               cones detected
#     """

#     separated_images = {"good": {}, "bad": {}}
#     # For each patch, separate them in good and bad images
#     # We first need to see which gaussian is corresponding to the good images
#     bad_label = np.argmin(gmm.means_)

#     for key, val in cones_detected.items():
#         label = gmm.predict([[val]])[0]
#         cone_image = re.sub("CROP_(\d+)_", "", key)
#         cone_image = re.sub("x(\d+)y(\d+)_", "", cone_image)
#         if label == bad_label :
#             if cone_image in separated_images["bad"].keys():
#                 separated_images["bad"][cone_image].append(key)
#             else :
#                 separated_images["bad"][cone_image] = [key]
#         else :
#             if cone_image in separated_images["good"].keys():
#                  separated_images["good"][cone_image].append(key)
#             else :
#                 separated_images["good"][cone_image] = [key]

#     return separated_images

# def separate_patches(cones_detected: dict, gmm: GaussianMixture) -> dict:
#     """ Separate patches from gaussian distributions """

#     separated_patches = {"good": [], "bad": []}
#     # For each patch, separate them in good and bad images
#     # We first need to see which gaussian is corresponding to the good images
#     bad_label = np.argmin(gmm.means_)
#     for key, val in cones_detected.items():
#         label = gmm.predict([[val]])[0]
#         if label == bad_label :
#             separated_patches["bad"].append(key)
#         else :
#             separated_patches["good"].append(key)
#     return separated_patches

# def find_overlapping_region(meaningful_idx: np.array, meaningful_idx_n: np.array) :
#     """Find the overlapping region between multiple images"""
#     # top = np.min(meaningful_idx[:, 0])
#     # bottom = np.max(meaningful_idx[:, 0])
#     # left = np.min(meaningful_idx[:, 1])
#     # right = np.max(meaningful_idx[:, 1])

#     # top_n = np.min(meaningful_idx_n[:, 0])
#     # bottom_n = np.max(meaningful_idx_n[:, 0])
#     # left_n = np.min(meaningful_idx_n[:, 1])
#     # right_n = np.max(meaningful_idx_n[:, 1])
#     meaningful_idx = np.reshape(meaningful_idx, (-1, meaningful_idx.shape[-1]))
#     meaningful_idx_n = np.reshape(meaningful_idx_n, (-1, meaningful_idx_n.shape[-1]))

#     # area1 = np.sum(meaningful_idx, axis=0)
#     # area2 = np.sum(meaningful_idx_n, axis=0)

#     # # intersections and union
#     # intersections = np.dot(meaningful_idx.T, meaningful_idx_n)
#     # union = area1[:, None] + area2[None, :] - intersections
#     # overlaps = intersections / union

#     return overlaps

# def move_calculated_split_images(input_folder: str) -> None :
#     """
#     Moves calculatedSplit images from subfolders in parent of input folder to input_folder

#     :param input_folder: directory where the images will be
#     """
#     os.makedirs(input_folder)
#     parent_dir = Path(input_folder).parent.absolute()
#     image_dirs = prepare_folders(parent_dir)
#     for image_dir in image_dirs :
#         for image in os.listdir(image_dir) :
#             if "CalculatedSplit" in image and "avg" in image :
#                 #import pdb
#                 #pdb.set_trace()
#                 #image_path = os.path.join(image_dir, image)
#                 shutil.copyfile(os.path.join(image_dir, image), os.path.join(input_folder, image))

