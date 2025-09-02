import enum

import numpy as np


FUNDUS_IMAGE_SIZES_IN_DEGREES = {
    'OCTA': [10, 10],
    'OCTA_HD': [20, 20],
    'HRA': [30, 30]
}

PIXEL_COORDINATES_DTYPE = np.int32

OPTIC_DISC_DISTANCE = 15.5

