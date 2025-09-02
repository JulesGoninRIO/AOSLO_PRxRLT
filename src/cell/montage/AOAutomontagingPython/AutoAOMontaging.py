import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import io, color
from skimage.transform import resize
from skimage.util import img_as_ubyte
import sys
# sys.path.append('.')
from .AOMosaicAllMultiModal import AOMosaicAllMultiModal  # Assuming you have this function defined somewhere

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
    # # result = func(*args, **kwargs)
    # # Measure memory
    # mem_usage = memory_usage((func, args, kwargs), interval=1, timeout=1)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # result = None
    # # Measure CPU
    # end_cpu = p.cpu_times()
    # cpu_usage_user = end_cpu.user - start_cpu.user
    # cpu_usage_system = end_cpu.system - start_cpu.system

    # return result, elapsed_time, max(mem_usage), cpu_usage_user, cpu_usage_system

def AutoAOMontaging(imgfolder_name, postionFile_name, outputFolder_name):
    """
    This function is a Python equivalent of the MATLAB function AutoAOMontaging.
    It performs similar operations but some MATLAB-specific functions are replaced with Python equivalents.
    """
    # Variables and defaults
    modalitiesInfo = [['Confocal', 'Confocal'],
                      ['Split Detection', 'CalculatedSplit'],
                      ['Dark Field', 'DarkField']]

    # Start timer
    start_time = time.time()

    # Call AOMosiacAllMultiModal function
    # ao = AOMosaicAllMultiModal(imgfolder_name, postionFile_name, outputFolder_name,
    #                            np.array(modalitiesInfo)[:,1].tolist())
    AOMosaicAllMultiModal(imgfolder_name, postionFile_name, outputFolder_name,
                          np.array(modalitiesInfo)[:,1].tolist()).run()
    # # Usage:
    # result, elapsed_time, max_mem_usage, cpu_usage_user, cpu_usage_system = measure_performance(ao.run)
    # print(f"Elapsed time: {elapsed_time} seconds")
    # print(f"Max memory usage: {max_mem_usage} MiB")
    # print(f"CPU usage: {cpu_usage_user} percent, system: {cpu_usage_system}")
    # print(result)

    # Print elapsed time
    print("Elapsed time: %s seconds" % (time.time() - start_time))