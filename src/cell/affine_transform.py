from typing import Tuple, Union, Self
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.affinity import affine_transform
from shapely.geometry import box, Polygon, Point, LineString

IMAGE_SIZE = 720

class AffineTransform:
    def __init__(self, matrix: np.array):
        if matrix.shape != (2, 3):
            raise ValueError("Matrix must be of shape (2, 3)")
        self.matrix = matrix
        self.translation = matrix[:2, 2]
        self.rotation = matrix[:2, :2]

    @classmethod
    def from_cv2(cls, matrix: np.array):
        """Create an affine transform from a cv2 transformation matrix."""
        return cls(matrix)

    @classmethod
    def from_translation(cls, dx, dy):
        """Create an affine transform that translates by (dx, dy)."""
        matrix = np.array([[1, 0, dx], [0, 1, dy]])
        return cls(matrix)

    # @classmethod
    # def from_scale(cls, sx, sy):
    #     """Create an affine transform that scales by (sx, sy)."""
    #     matrix = np.array([[sx, 0, 0], [0, sy, 0]])
    #     return cls(matrix)

    @classmethod
    def from_rotation(cls, theta):
        """Create an affine transform that rotates by theta radians."""
        matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0]])
        return cls(matrix)

    def combine(self, other: Self) -> Self:
        # Usage:
        # translation = AffineTransform.from_translation(dx, dy)
        # rotation = AffineTransform.from_rotation(theta)
        # combined = translation.combine(rotation)
        """Combine this transform with another one by matrix multiplication."""
        combined_matrix = np.dot(self.matrix, np.vstack((other.matrix, [0, 0, 1])))
        return AffineTransform(combined_matrix[:2])

    def translation_distance(self, other: Self, ratio: Tuple[float, float] = (1, 1)):
            scaled_self_translation = self.translation * ratio
            scaled_other_translation = other.translation * ratio
            return np.linalg.norm(scaled_self_translation - scaled_other_translation)

    def rotation_distance(self, other: Self):
        return  np.linalg.norm(R.from_matrix(np.pad(self.rotation, ((0, 1), (0, 1)), 'constant')).as_euler('xyz') - R.from_matrix(np.pad(other.rotation, ((0, 1), (0, 1)), 'constant')).as_euler('xyz'))

    def compute_distance(self, other: Self):
        return self.translation_distance(other) + self.rotation_distance(other)

    # def compute_top_left_distance(self, other: Self):
    #     return self.translation_distance(other)

    def compute_overlap_region(self, other: Self, ratio = (1,1), shape: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> Union[Polygon, Point, LineString]:
        box1 = box(0, 0, shape[0], shape[1])
        box2 = box(0, 0, shape[0], shape[1])
        this_matrix = adapt_transform_for_new_size(self.matrix, ratio)
        other_matrix = adapt_transform_for_new_size(other.matrix, ratio)
        transformed_box1 = affine_transform(box1, this_matrix.flatten(order='F').tolist())
        transformed_box2 = affine_transform(box2, other_matrix.flatten(order='F').tolist())
        overlap_region = transformed_box1.intersection(transformed_box2)
        return overlap_region

    def transform_points(self, points: np.ndarray, ratio = (1,1)) -> np.ndarray:
        if points.ndim == 1:
            points = points.reshape(1, -1)
        # points = np.hstack((points, np.ones((points.shape[0], 1))))
        matrix = adapt_transform_for_new_size(self.matrix, ratio)
        transform = np.vstack((matrix, [0, 0, 1]))

        import cv2
        points_transformed = cv2.perspectiveTransform(points.astype(np.float32).reshape(1, -1, 2), transform.astype(np.float32))

        # points_transformed = (transform[:2] @ points.T).T
        return points_transformed[0]

    def _is_greater_in_coordinate(self, other: Self, index: int) -> bool:
        point_element_transformed = self.transform_points(np.array([[0, 0]]))
        point_neighbor_transformed = other.transform_points(np.array([[0, 0]]))
        return point_neighbor_transformed[0][index] > point_element_transformed[0][index]

    def is_right(self, other: Self) -> bool:
        return self._is_greater_in_coordinate(other, 0)

    def is_below(self, other: Self) -> bool:
        return self._is_greater_in_coordinate(other, 1)

def adapt_transform_for_new_size(matrix, ratio: Tuple[int, int]):
    # TODO: test and maybe include somewhere else
    # Calculate the scale ratios for width and height
    scale_ratio_width, scale_ratio_height = ratio

    # Adjust the scale components of the matrix
    # Assuming the scale is uniformly applied, we use the average of the width and height ratios
    scale_ratio = (scale_ratio_width + scale_ratio_height) / 2
    scaled_matrix = matrix.copy().astype(np.float64)
    scaled_matrix[0, :] *= scale_ratio
    scaled_matrix[1, :] *= scale_ratio

    # Adjust the translation components
    # This step depends on how the translation should adapt to the new size.
    # For simplicity, we scale the translation components by the same scale ratio.
    # This might need adjustment based on specific requirements.
    # scaled_matrix[0, 2] *= scale_ratio_width
    # scaled_matrix[1, 2] *= scale_ratio_height
    return scaled_matrix
