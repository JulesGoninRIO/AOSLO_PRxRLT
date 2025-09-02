from enum import Enum, unique
from typing import Self

@unique
class Direction(Enum):
    """
    Enum to represent directions X and Y.

    Example:
    >>> for direction in Direction:
    ...     print(direction, direction.value, direction.is_X)
    Direction.X X True
    Direction.Y Y False
    """

    X = 'X'
    @property
    def is_X(self) -> bool:
        return self == Direction.X

    Y = 'Y'
    @property
    def is_Y(self) -> bool:
        return self == Direction.Y

    def __iter__(self):
        """
        Allows for iteration over all the directions.
        """
        return iter(Direction.__members__.values())

    @classmethod
    def from_str(cls, name: str) -> Self | None:
        """
        Parse a string to a Direction enum value.

        :param value: The string to convert.
        :type value: str
        :return: The Direction enum value.
        :rtype: Direction
        """
        name = name.upper()
        for drct_name, dirct in Direction.__members__.items():
            if name == drct_name or name.startswith(drct_name) \
                                 or name.endswith(drct_name) \
                                 or '_' + drct_name in name:
                return dirct
        # could not parse given string
        return None