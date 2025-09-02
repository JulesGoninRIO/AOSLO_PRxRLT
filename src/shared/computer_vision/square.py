class Square:
    def __init__(self, top_left: int, top_right: int, bottom_left: int, bottom_right: int):
        """
        Initialize a Square object with the given corner points.

        :param top_left: The top-left corner point.
        :type top_left: int
        :param top_right: The top-right corner point.
        :type top_right: int
        :param bottom_left: The bottom-left corner point.
        :type bottom_left: int
        :param bottom_right: The bottom-right corner point.
        :type bottom_right: int
        """
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
