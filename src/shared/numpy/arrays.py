from typing import List, Tuple
import numpy as np
import cv2


from src.shared.helpers.intervals import adjust_intervals

def mask_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to an array, setting elements to zero where the mask is False.

    This function takes an array and a mask of the same shape, and sets the elements
    of the array to zero where the mask is False.

    :param arr: The input array to be masked.
    :type arr: np.ndarray
    :param mask: The mask array, where True indicates the elements to keep.
    :type mask: np.ndarray
    :return: The masked array with elements set to zero where the mask is False.
    :rtype: np.ndarray
    :raises AssertionError: If the shapes of the array and mask do not match.
    :raises np.core._exceptions._ArrayMemoryError: If there is not enough memory to perform the operation.
    """
    assert arr.shape == mask.shape, "Array and mask must have the same shape"
    arr_cut = arr.copy()
    try:
        arr_cut[~mask] = 0
    except np.core._exceptions._ArrayMemoryError:
        # we are running out of Memory
        raise
    return arr_cut

def get_filled_pixels_corners(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the corners of the filled pixels in a mask.

    This function finds the top, bottom, left, and right coordinates of the filled pixels
    (i.e., where the mask is True) in a binary mask array.

    :param mask: The binary mask array.
    :type mask: np.ndarray
    :return: A tuple containing the top, bottom, left, and right coordinates of the filled pixels.
    :rtype: Tuple[int, int, int, int]
    :raises ValueError: If there are no filled pixels in the mask.
    """
    filled_pixels = np.argwhere(mask)
    if filled_pixels.shape[0] == 0:
        return None
    try:
        top = np.min(filled_pixels[:, 0])
        bottom = np.max(filled_pixels[:, 0])
        left = np.min(filled_pixels[:, 1])
        right = np.max(filled_pixels[:, 1])
    except ValueError:
        return None
    return top, bottom, left, right

def pad_area_both_directions(area: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Pad the given area based on the given indices.

    :param area: the area to be padded
    :type area: np.ndarray
    :param i: the y direction pixel shift
    :type i: int
    :param j: the x direction pixel shift
    :type j: int
    :return: the padded area
    :rtype: np.ndarray
    """
    pad_width = ((max(0, i), max(0, i)), (max(0, j), max(0, j)))
    return np.pad(area, pad_width, 'constant')


def adjust_array(img: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """
    Adjust the array by padding it with zeros if necessary.

    This function adjusts the input array by padding it with zeros if the specified top, bottom,
    left, or right indices are out of bounds.

    :param img: The input array to be adjusted.
    :type img: np.ndarray
    :param top: The top index to adjust.
    :type top: int
    :param bottom: The bottom index to adjust.
    :type bottom: int
    :param left: The left index to adjust.
    :type left: int
    :param right: The right index to adjust.
    :type right: int
    :return: The adjusted array with necessary padding.
    :rtype: np.ndarray
    """
    if top < 0:
        img = np.pad(img, ((-top, 0), (0, 0)), 'constant')
    if left < 0:
        img = np.pad(img, ((0, 0), (-left, 0)), 'constant')
    if bottom > img.shape[0]:
        img = np.pad(img, ((0, bottom - img.shape[0]), (0, 0)), 'constant')
    if right > img.shape[1]:
        img = np.pad(img, ((0, 0), (0, right - img.shape[1])), 'constant')
    return img

def adjust_and_pad_arrays(
    img1_cut: np.ndarray,
    interval1_end: int,
    img2_cut: np.ndarray,
    interval2_end: int,
    axis: int) -> Tuple[np.ndarray, int, np.ndarray, int]:
    """
    Adjust and pad two arrays to make their intervals equal.

    This function adjusts and pads two input arrays along the specified axis so that their intervals
    are equal in length.

    :param img1_cut: The first input array to be adjusted and padded.
    :type img1_cut: np.ndarray
    :param interval1_end: The end index of the interval for the first array.
    :type interval1_end: int
    :param img2_cut: The second input array to be adjusted and padded.
    :type img2_cut: np.ndarray
    :param interval2_end: The end index of the interval for the second array.
    :type interval2_end: int
    :param axis: The axis along which to adjust and pad the arrays (0 for rows, 1 for columns).
    :type axis: int
    :return: A tuple containing the adjusted and padded arrays and their new interval ends.
    :rtype: Tuple[np.ndarray, int, np.ndarray, int]
    """
    if interval2_end > interval1_end:
        img1_cut = img1_cut[:interval1_end, :] if axis == 0 else img1_cut[:, :interval1_end]
        img1_cut = np.pad(img1_cut, ((0, interval2_end - interval1_end), (0, 0)) if axis == 0 else ((0, 0), (0, interval2_end - interval1_end)), 'constant')
        interval1_end = interval2_end
    elif interval1_end > interval2_end:
        img2_cut = img2_cut[:interval2_end, :] if axis == 0 else img2_cut[:, :interval2_end]
        img2_cut = np.pad(img2_cut, ((0, interval1_end - interval2_end), (0, 0)) if axis == 0 else ((0, 0), (0, interval1_end - interval2_end)), 'constant')
        interval2_end = interval1_end
    return img1_cut, interval1_end, img2_cut, interval2_end


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping

    :param image: the image to rotate
    :type image: np.ndarray
    :param angle: the angle of rotation
    :type angle: float
    :return: the rotated image
    :rtype: np.ndarray
    """

    height, width = image.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = round(height * abs_sin + width * abs_cos)
    bound_h = round(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))

    return rotated_image

def get_boundaries(dst: np.ndarray) -> List[int]:
    """
    Get the boundaries where the image pixels are

    :param dst: the image
    :return: list that has the form [top, bottom, left, right]

    :param dst: the image with the two modalities overlapping
    :type dst: np.ndarray
    :return: list that has the form [top, bottom, left, right]
    :rtype: List[int]
    """

    filled_pixels = np.argwhere(dst > 0)
    top = np.min(filled_pixels[:, 0])
    bottom = np.max(filled_pixels[:, 0])
    left = np.min(filled_pixels[:, 1])
    right = np.max(filled_pixels[:, 1])

    return [top, bottom, left, right]

def zero_padding(src: np.ndarray, shape: Tuple[int, int], pos: np.array) -> np.ndarray:
    """
    Add zeros on the border of the images

    :param src: source image to pad
    :type src: np.ndarray
    :param shape: the shape to be padded to
    :type shape: Tuple[int, int]
    :param pos: the position y and x where the image will be
    :type pos: np.array
    :return: the image padded
    :rtype: np.ndarray
    """

    y, x = (int(pos[0]), int(pos[1]))
    padded_image = np.zeros(shape)
    padded_image[y:src.shape[0] + y, x:src.shape[1] + x] = src

    return padded_image

