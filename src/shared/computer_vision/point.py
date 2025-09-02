import math
from typing import Tuple, Self
import numpy as np

class Point:
    def __init__(self, x: int = 0, y: int = 0):
        """
        Initialize a Point object.

        :param x: The x-coordinate of the point.
        :type x: int
        :param y: The y-coordinate of the point.
        :type y: int
        """
        self.x = x
        self.y = y

    def round(self) -> Self:
        """
        Round the coordinates of the point to the nearest integer.

        :return: A new Point object with the rounded coordinates.
        :rtype: Point
        """
        return Point(round(self.x), round(self.y))

    def __iter__(self) -> iter:
        """
        Return an iterator over the coordinates of the point.

        :return: An iterator over the coordinates of the point.
        :rtype: iter
        """
        return iter((self.x, self.y))
    
    def to_np(self) -> np.ndarray:
        """
        Convert the point to a NumPy array.

        :return: A NumPy array representing the point.
        :rtype: np.ndarray
        """
        return np.array([self.x, self.y])

    @classmethod
    def from_cartesian(cls, x: int, y: int):
        """
        Create a Point object from Cartesian coordinates.

        :param x: The x-coordinate of the point.
        :type x: int
        :param y: The y-coordinate of the point.
        :type y: int
        :return: A Point object.
        :rtype: Point
        """
        return cls(x, y)

    @classmethod
    def from_polar(cls, r: int, theta: int):
        """
        Create a Point object from polar coordinates.

        :param r: The radius (distance from the origin).
        :type r: int
        :param theta: The angle in radians.
        :type theta: int
        :return: A Point object.
        :rtype: Point
        """
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return cls(x, y)
