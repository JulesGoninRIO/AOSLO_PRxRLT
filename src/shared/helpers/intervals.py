from typing import Tuple

def adjust_intervals(
    interval1_start: int,
    interval1_end: int,
    interval2_start: int,
    interval2_end: int) -> Tuple[int, int, int, int]:
    """
    Adjust intervals to make them equal in length.

    This function adjusts the start and end points of two intervals so that they are equal in length.
    If the second interval is longer, the first interval is adjusted to match the second interval.
    If the first interval is longer, the second interval is adjusted to match the first interval.

    :param interval1_start: The start point of the first interval.
    :type interval1_start: int
    :param interval1_end: The end point of the first interval.
    :type interval1_end: int
    :param interval2_start: The start point of the second interval.
    :type interval2_start: int
    :param interval2_end: The end point of the second interval.
    :type interval2_end: int
    :return: A tuple containing the adjusted start and end points of both intervals.
    :rtype: Tuple[int, int, int, int]
    """
    if interval2_end - interval2_start > interval1_end - interval1_start:
        interval1_start = interval2_start
        interval1_end = interval2_end
    if interval1_end - interval1_start > interval2_end - interval2_start:
        interval2_start = interval1_start
        interval2_end = interval1_end
    return interval1_start, interval1_end, interval2_start, interval2_end