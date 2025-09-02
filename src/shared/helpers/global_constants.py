import numpy as np

PIXEL_COORDINATES_DATATYPE = np.int16
"""
Datatype in which we store the cell coordinates in pixels.
Since the size of the images is unlikely to exceed 32000x32000, 16-bit int is
enough.
"""

DEGREE_COORDINATES_DATATYPE = np.float64
"""
Datatype in which we store the cell coordinates in degrees.
Since we need only 2-3 digits after comma, single precision is enough.
"""

EMPTY_PIXEL_COORDINATES = np.empty((0, 2), dtype=PIXEL_COORDINATES_DATATYPE)
"""
Shortcut for the empty pixel coordinates array, e.g. for initialization or
returning in case of an error.
"""

NON_EXISTENT_VERTEX = -1
"""
If a ridge vertex of a voronoi diagram takes this value, it lies beyond the
image border.
"""

GOOD_NUMBER_OF_NEIGHBORS = 6
"""
We consider hexagonal cell grid as a perfect one, and refine the cell
coordinates according to it.
"""

MDRNN_PATCH_SIZE = 185
"""
Size of the images to run through Davidson's algorithm
"""

IMAGE_SIZE = 1080 # 720
"""
Size of the images.
"""

MDRNN_OVERLAP_DISTANCE_SAME_IMAGE = 5
"""
Maximal distance between cones to merge them as a single cone when reconstructing
the images from patches.
"""

MDRNN_OVERLAP_DISTANCE = 5
"""
Maximal distance between cones to merge them as a single cone when reconstructing
the images from patches.
"""

ATMS_OVERLAP_DISTANCE = 3
"""
Maximal distance between cones to merge them as a single cone when reconstructing
the images from patches.
"""

NIEGHBOR_CIRCULAR_SHAPE = True
"""
Whether you want to search for neighbors in a squared distance or circular distance.
"""

class _ExponentialAverage:
    """
    Class to handle the exponential average ???
    """

    def __init__(self, averaging_factor: float = None):
        """
        Initialize the class with None value and average factor given

        :param averaging_factor: the average factor number
        """
        self.__value = None
        self._averaging_factor = averaging_factor

    def __get__(self, instance, owner):
        return self.__value

    def __set__(self, instance, value):
        """
        Average the given value with the value already stored in this object. If
        it does not exist yet, initialize it with the given value.
        """

        if self.__value is None or value is None:
            self.__value = value
        else:
            if self._averaging_factor is not None:
                self.__value = self._averaging_factor * value + (1.0 - self._averaging_factor) * self.__value
            else:
                global_parameters = GlobalParameterContainer()
                self.__value = global_parameters.voronoi_threshold_averaging_factor * value + \
                               (1.0 - global_parameters.voronoi_threshold_averaging_factor) * self.__value


class Singleton(type):
    """
    Only one instance of this class can exist at a time. All constructors will
    return the same instance, which can be used to access and modify its fields
    in any part of the program.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalThresholdContainer(metaclass=Singleton):
    """
    Singleton class that stores some global parameters. They can be accessed and
    modified through any instance of the class, since they are the same for all
    of them.
    """

    removing_distance_threshold = _ExponentialAverage()
    removing_luminance_threshold = _ExponentialAverage()
    addition_distance_threshold = _ExponentialAverage()
    dark_areas_threshold = _ExponentialAverage()

    def __init__(self):
        pass

    def reset(self) -> None:
        self.removing_distance_threshold = None
        self.removing_luminance_threshold = None
        self.addition_distance_threshold = None
        self.dark_areas_threshold = None


class GlobalParameterContainer(metaclass=Singleton):
    """
    Class that contains the global parameters
    """

    def __init__(
            self,
            min_cell_distance: int = 1,
            thresholding_window_size: int = 15,
            voronoi_threshold_averaging_factor: float = 0.03,
            dark_area_std_multiplier: float = 1.2,
            refinement_removing_luminosity_std_multiplier: float = 1.5,
            refinement_addition_luminosity_std_multiplier: float = 0.0
    ):
        self.min_cell_distance = min_cell_distance
        """Minimum possible distance in pixels btw two cells.
        In principle, it depends on the resolution of the images."""

        self.thresholding_window_size = thresholding_window_size
        """Size of the window inside which the same thresholding value is applied."""

        self.voronoi_threshold_averaging_factor = voronoi_threshold_averaging_factor
        """Coefficient of the exponential average of the cell distance thresholds."""

        self.dark_area_std_multiplier = dark_area_std_multiplier
        """It is used in adaptive detection of shadowed areas. The threshold value for it is mu - DASM * std, and the
        lower is DASM the less tolerance there will be for shadowing."""

        self.refinement_removing_luminosity_std_multiplier = refinement_removing_luminosity_std_multiplier
        """When the initially predicted cells are refined, it is used to remove dark cells."""

        self.refinement_addition_luminosity_std_multiplier = refinement_addition_luminosity_std_multiplier
        """When the initially predicted cells are refined, it is used to not to add the cells back."""

    def set(
            self,
            min_cell_distance: int = None,
            thresholding_window_size: int = None,
            voronoi_threshold_averaging_factor: float = None,
            dark_area_std_multiplier: float = None,
            refinement_removing_luminosity_std_multiplier: float = None,
            refinement_addition_luminosity_std_multiplier: float = None
    ) -> None:

        if min_cell_distance is not None:
            self.min_cell_distance = min_cell_distance

        if thresholding_window_size is not None:
            self.thresholding_window_size = thresholding_window_size

        if voronoi_threshold_averaging_factor is not None:
            self.voronoi_threshold_averaging_factor = voronoi_threshold_averaging_factor

        if dark_area_std_multiplier is not None:
            self.dark_area_std_multiplier = dark_area_std_multiplier

        if refinement_removing_luminosity_std_multiplier is not None:
            self.refinement_removing_luminosity_std_multiplier = refinement_removing_luminosity_std_multiplier

        if refinement_addition_luminosity_std_multiplier is not None:
            self.refinement_addition_luminosity_std_multiplier = refinement_addition_luminosity_std_multiplier
