import enum

import numpy as np


IMAGE_DIR = r'T:\Studies\AOSLO-2021\AST\Sample gallery'

# AOSLO_IMAGE_SIZE_IN_DEGREES = np.array([1.5, 1.5])
AOSLO_IMAGE_SIZE_IN_DEGREES = np.array([3., 3.])


FUNDUS_IMAGE_SIZES_IN_DEGREES = {
    'OCTA': [10, 10],
    'OCTA_HD': [20, 20],
    'HRA': [30, 30]
}

PIXEL_COORDINATES_DTYPE = np.int32

MAX_AOSLO_ECCENTRICITY = 13

OPTIC_DISC_DISTANCE = 15.5


@enum.unique
class Indices(enum.IntEnum):
    FUNDUS_IND = 0
    ZOOMED_IND = 1
