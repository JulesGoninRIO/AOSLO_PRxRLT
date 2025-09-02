import numpy as np
from typing import Tuple, List
import cv2
import logging

from src.cell.affine_transform import AffineTransform

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

def warp_image(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply affine transform to the image.

    :param image: The input image to be transformed.
    :type image: np.ndarray
    :param H: The affine transformation matrix.
    :type H: np.ndarray
    :return: The transformed image.
    :rtype: np.ndarray
    """
    h, w = image.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1] / map_ind[-1]
    return cv2.remap(
        image, map_x.reshape(h, w).astype(np.float32),
        map_y.reshape(h, w).astype(np.float32), cv2.INTER_LINEAR
    )

def blend_images(modality_image: np.ndarray, oa_image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Blend two images with given offsets.

    :param modality_image: The first image to blend.
    :type modality_image: np.ndarray
    :param oa_image: The second image to blend.
    :type oa_image: np.ndarray
    :param H: The affine transformation matrix.
    :type H: np.ndarray
    :return: The blended image.
    :rtype: np.ndarray
    :raises ValueError: If either modality_image or oa_image is not 2-dimensional.
    """
    if modality_image.ndim != 2 or oa_image.ndim != 2:
        raise ValueError("Both modality_image and oa_image must be 2-dimensional")
    dy = H[1, 2]
    dx = H[0, 2]
    blended_shape = (
        modality_image.shape[0] + np.abs(round(dy)),
        modality_image.shape[1] + np.abs(round(dx)), 3)
    blended = np.zeros(blended_shape)
    oa_offset_x, oa_offset_y = max(0, round(dx)), max(0, round(dy))
    mod_offset_x, mod_offset_y = max(0, -round(dx)), max(0, -round(dy))
    blended[
        oa_offset_y:oa_offset_y + oa_image.shape[0],
        oa_offset_x:oa_offset_x + oa_image.shape[1], 2] += 0.5 * oa_image
    blended[
        mod_offset_y:mod_offset_y + modality_image.shape[0],
        mod_offset_x:mod_offset_x + modality_image.shape[1], 1] += 0.5 * modality_image
    return blended

def get_blended_image(
    H: AffineTransform,
    first_image: np.ndarray,
    second_image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Get the blended image from two input images using an affine transformation.

    :param H: The affine transformation object.
    :type H: AffineTransform
    :param first_image: The first image to blend.
    :type first_image: np.ndarray
    :param second_image: The second image to blend.
    :type second_image: np.ndarray
    :return: A tuple containing the blended image and the boundaries.
    :rtype: Tuple[np.ndarray, List[str]]
    """
    dst = warp_image(second_image, H.matrix)
    boundaries = get_boundaries(dst)
    blended = blend_images(first_image, dst, H.matrix)
    return blended, boundaries

def shift_image(p1: Tuple[int, int], p2: Tuple[int, int],
                image: np.ndarray = None) -> Tuple[np.ndarray, List[int]]:
    """
    Shifts an image from top left and top right corner of the image

    :param p1: top left of the image
    :type p1: Tuple[int, int]
    :param p2: top right of the image
    :type p2: Tuple[int, int]
    :param image: the image to shift, defaults to None
    :type image: np.ndarray, optional
    :return: the image shifted and its transform
    :rtype: Tuple[np.ndarray, List[int]]
    """

    angle_image = -angle_between(p1, p2)
    if not np.any(image):
        image = np.ones((IMAGE_SIZE, IMAGE_SIZE))
        image_own_angle = np.array(Image.fromarray(image).rotate(angle_image,
                                                                 expand=1))

    elif len(image.shape) < 3:
        only_ones = np.ones(image.shape)
        image_own_angle = np.array(Image.fromarray(image).rotate(angle_image,
                                                                 expand=1))
        alpha_image = np.array(Image.fromarray(only_ones).rotate(angle_image,
                                                                 expand=1))
        image_own_angle = np.transpose(
            np.stack((image_own_angle, alpha_image)), (1, 2, 0))

    # # let's verify the dimensions after the rotation and correct for zero padding
    # # around the image
    try:
        top, bottom, left, right = get_boundaries(image_own_angle[:, :, 1])
        image_own_angle = image_own_angle[top:bottom+1, left:right+1, :]
    except IndexError:
        top, bottom, left, right = get_boundaries(image_own_angle)
        image_own_angle = image_own_angle[top:bottom+1, left:right+1]

    # correct for the rotation of the image if it has an angle
    if angle_image > 0:
        # if angle positive, rotate anti-clockwise -> added zeros in the y direction
        # left is more important
        try:
            top_left = np.argwhere(image_own_angle[:, :, 1])[
                np.argmin(np.argwhere(image_own_angle[:, :, 1])[:, 1])]
        except IndexError:
            top_left = np.argwhere(image_own_angle)[
                np.argmin(np.argwhere(image_own_angle)[:, 1])]
        translation = p1 - top_left
    elif angle_image < 0:
        # if angle negative, rotate clockwise -> added zeros in the x direction
        try:
            top_left = np.argwhere(image_own_angle[:, :, 1])[
                np.argmin(np.argwhere(image_own_angle[:, :, 1])[:, 0])]
        except IndexError:
            top_left = np.argwhere(image_own_angle)[
                np.argmin(np.argwhere(image_own_angle)[:, 0])]
        translation = p1 - top_left
    else:
        translation = p1

    # pad to upper left until correct shape
    if translation[0] < 0:
        logging.info(f"translation[0] smaller than 0 {translation[0]}")
        translation[0] = 0
    if translation[1] < 0:
        logging.info(f"translation[1] smaller than 0 {translation[1]}")
        translation[1] = 0
    try:
        if len(image_own_angle.shape) > 2:
            image = np.pad(image_own_angle, (
                (round(translation[0]), 0), (round(translation[1]), 0), (0, 0)
                ), 'constant')
        else:
            image = np.pad(image_own_angle, (
                (round(translation[0]), 0), (round(translation[1]), 0)
                ), 'constant')
    except ValueError:
        logging.error("Cannot shift the image")
        raise

    return image, translation
