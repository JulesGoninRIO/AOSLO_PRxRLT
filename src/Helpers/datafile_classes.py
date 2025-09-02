import csv
import logging
import os
import re
import warnings
from enum import Enum, IntEnum, auto, unique
from math import floor
from typing import Any, Callable, Dict, List, Optional, Union

# newest versions of OpenCV changed the way of importing
try:
    from cv2 import cv2
except ImportError:
    import cv2

import numpy as np

from .global_constants import (DEGREE_COORDINATES_DATATYPE,
                               EMPTY_PIXEL_COORDINATES,
                               PIXEL_COORDINATES_DATATYPE)
from .validators import (_FLOAT_PATTERN, _INT_PATTERN, _LETTER_AND_INT_PATTERN,
                         _LETTER_ONLY_PATTERN, _STRING_PATTERN, _ParameterType,
                         _SupportedTypes, _validate_value)

@unique
class ImageModalities(Enum):
    CO = 'Confocal'
    CS = 'CalculatedSplit'
    OA = 'OA850nm'  # or 'OA850' or '850nm' or 'OA'
    DF = 'DarkField'


IMAGE_EXTENSIONS = ['png', 'tif']
COORDINATES_EXTENSIONS = ['csv', 'txt']

_DEFAULT_VALUES: Dict[int, _ParameterType] = {
    _SupportedTypes.INT: float('inf'),
    _SupportedTypes.POSITIVE_INT: -1,
    _SupportedTypes.FLOAT: float('inf'),
    _SupportedTypes.POSITIVE_FLOAT: -1.,
    _SupportedTypes.STRING: '__None__',
    _SupportedTypes.LETTER_ONLY_STRING: 'None'
}
"""if the datafile parameters are not initialized, they will take these values"""


def _default_value(value_type: _SupportedTypes) -> _ParameterType:
    """
    :param value_type: id of one of the available types
    :return: the default value for the parameter of the given type
    :raises ValueError: if the given ID does not correspond to any of the
                        supported types
    """

    if value_type not in _DEFAULT_VALUES:
        raise ValueError('Default value not defined')

    return _DEFAULT_VALUES[value_type]


class DataFile:
    """
    This base class envelopes some useful methods for data files used in the
    project.
    """

    def _property_getter(self, name: str, value_type: _SupportedTypes) -> _ParameterType:
        """
        All the properties of the class have the same pattern of getter. First,
        the value is validated.
        Second, if the value is the default one, then the property has not been
        initialized, and an error is raised.
        And finally, if everything is OK, the value is returned.
        This function implements this pattern taking validators and default
        values corresponding to the desired property.

        :param name: name of the property being accessed
        :param value_type: id of the type of the expected value for this
                           property (field of _SupportedTypes)
        :return: the value of the property
        """

        attribute_value = self.__getattribute__(f'_{name}')
        if attribute_value != _default_value(value_type):
            return attribute_value
        else:
            raise RuntimeError(f'{name} is not initialized')

    def _property_setter(
            self,
            name: str,
            value_type: _SupportedTypes,
            value: _ParameterType) -> None:
        """
        All the properties of the class have the same pattern of getter.
        First, if None provided as the value, the corresponding default value is
        assigned to the field. Otherwise, the given value is validated, and if
        correct, assigned to the field. This function implements this pattern
        taking validators and default values corresponding to the desired
        property.

        :param name: name of the property being accessed
        :param value_type: id of the type of the expected value for this
                           property (field of _SupportedTypes)
        :param value: the value of the property
        """

        attribute_name = f'_{name}'
        if value is not None:
            _validate_value(value_type, value, type(
                self).__name__ + f'.{name}')
            self.__setattr__(attribute_name, value)
        else:
            self.__setattr__(attribute_name, _default_value(value_type))

    @property
    def prefix(self) -> str:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('prefix', _SupportedTypes.STRING)

    @prefix.setter
    def prefix(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter('prefix', _SupportedTypes.STRING, value)

    @property
    def subject_id(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the value does not belong to a valid range for the desired type
        """
        return self._property_getter('subject_id', _SupportedTypes.POSITIVE_INT)

    @subject_id.setter
    def subject_id(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value is negative
        """
        self._property_setter(
            'subject_id', _SupportedTypes.POSITIVE_INT, value)

    @property
    def session_id(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('session_id', _SupportedTypes.POSITIVE_INT)

    @session_id.setter
    def session_id(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value is negative
        """
        self._property_setter(
            'session_id', _SupportedTypes.POSITIVE_INT, value)

    @property
    def eye(self) -> str:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('eye', _SupportedTypes.LETTER_ONLY_STRING)

    @eye.setter
    def eye(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value contains any symbols other from letters
        """
        self._property_setter('eye', _SupportedTypes.LETTER_ONLY_STRING, value)

    @property
    def x_position(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('x_position', _SupportedTypes.INT)

    @x_position.setter
    def x_position(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter('x_position', _SupportedTypes.INT, value)

    @property
    def y_position(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('y_position', _SupportedTypes.INT)

    @y_position.setter
    def y_position(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter('y_position', _SupportedTypes.INT, value)

    @property
    def x_size(self) -> float:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('x_size', _SupportedTypes.POSITIVE_FLOAT)

    @x_size.setter
    def x_size(self, value: float) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value is negative
        """
        self._property_setter('x_size', _SupportedTypes.POSITIVE_FLOAT, value)

    @property
    def y_size(self) -> float:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('y_size', _SupportedTypes.POSITIVE_FLOAT)

    @y_size.setter
    def y_size(self, value: float) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value is negative
        """
        self._property_setter('y_size', _SupportedTypes.POSITIVE_FLOAT, value)

    @property
    def image_id(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('image_id', _SupportedTypes.POSITIVE_INT)

    @image_id.setter
    def image_id(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value is negative
        """
        self._property_setter('image_id', _SupportedTypes.POSITIVE_INT, value)

    @property
    def modality(self) -> str:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('modality', _SupportedTypes.STRING)

    @modality.setter
    def modality(self, value: str) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value contains any symbols other from letters
        """
        self._property_setter('modality', _SupportedTypes.STRING, value)

    @property
    def postfix(self) -> str:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('postfix', _SupportedTypes.STRING)

    @postfix.setter
    def postfix(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter('postfix', _SupportedTypes.STRING, value)

    @property
    def extension(self) -> str:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('extension', _SupportedTypes.LETTER_ONLY_STRING)

    @extension.setter
    def extension(self, value: int) -> None:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises TypeError: if the value has a wrong type
        :raises ValueError: if the value contains any symbols other from letters
        """
        self._property_setter(
            'extension', _SupportedTypes.LETTER_ONLY_STRING, value)

    @property
    def crop_id(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('crop_id', _SupportedTypes.INT)

    @crop_id.setter
    def crop_id(self, value: int) -> None:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises TypeError: if the value has a wrong type
        """
        # print(self._property_getter('crop_id', _SupportedTypes.INT), value)
        self._property_setter('crop_id', _SupportedTypes.INT, value)
        # print(self._property_getter('crop_id', _SupportedTypes.INT))

    @property
    def crop_x_position(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('crop_x_position', _SupportedTypes.POSITIVE_INT)

    @crop_x_position.setter
    def crop_x_position(self, value: int) -> None:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter(
            'crop_x_position', _SupportedTypes.POSITIVE_INT, value)

    @property
    def crop_y_position(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('crop_y_position', _SupportedTypes.POSITIVE_INT)

    @crop_y_position.setter
    def crop_y_position(self, value: int) -> None:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter(
            'crop_y_position', _SupportedTypes.POSITIVE_INT, value)

    @classmethod
    def make_copy(cls, datafile):
        new_datafile = DataFile()
        new_datafile.__dict__ = datafile.__dict__
        new_datafile.__class__ = datafile.__class__
        return new_datafile

    def __init__(self, filename: str = None, datafile=None):
        """
        If neither filename not datafile provided, the parameters will be initialized with None.

        If the datafile is given, then the all its fields will be copied into this object. But be careful about the
        file extensions: they are not changed automatically, so if an instance of a subclass is copied into another
        subclass, data reading problems might occur.

        Otherwise, the filename must be given in the following format:

        '{}Subject{}_Session{}_{}_({},{})_{}x{}_{}_{}{}.{}'

        corresponding to the following parameters:
        1. prefix - optional string, if given, then must be in the form 'CROP_{}_x{}y{}_', with parameters
            - crop_id - positive int
            - crop_x_position - positive int
            - crop_y_position - positive int
        2. subject_id - positive int
        3. session_id - positive int
        4. eye - letter-only string; basically, should be one of the 'OS', 'OD',
            but in fact no restrictions are laid on it
        5. x_position - int
        6. y_position - int
        7. x_size - float
        8. y_size - float
        9. image_id - positive int
        10. modality - string; basically, should be one of the 'CalculatedSplit', 'Confocal',
            'Direct', 'OA850nm', ... but in fact no restrictions are laid on it
        11. postfix - optional string, can be anything
        12. extension - letter-only string, expected to be an extension of a real file

        Example:
        'CROP_1_x180y180_Subject34_Session274_OS_(-10,0)_1.5x1.5_4506_CalculatedSplit_extract_reg_avg.tif'

        'CROP_1_x430y210_Subject34_Session274_OS_(-1,-1)_1.5x1.5_4530_CalculatedSplit_extract_reg_avg.tif.png'

        'CROP_1_x430y210_Subject34_Session274_OS_(-1,-1)_1.5x1.5_4530_CalculatedSplit_extract_reg_avg.tif.csv'

        'Subject34_Session274_OS_(-10,0)_1.5x1.5_4506_CalculatedSplit_extract_reg_avg.tif'

        :raise ValueError: if the given filename cannot be parsed according to the given scheme.
        """

        if datafile is not None:
            self.__dict__ = datafile.__dict__.copy()
            return

        if filename is not None:
            # parse the filename onto common parameters
            parameters = re.search(
                r'({})Subject({})_Session({})_({})_\(({}),({})\)_({})x({})_({})_({})({})\.({})$'.format(
                    _STRING_PATTERN, _INT_PATTERN, _INT_PATTERN, _LETTER_ONLY_PATTERN, _INT_PATTERN, _INT_PATTERN,
                    _FLOAT_PATTERN, _FLOAT_PATTERN, _INT_PATTERN, _LETTER_AND_INT_PATTERN, _STRING_PATTERN,
                    _LETTER_ONLY_PATTERN
                ), filename
            )

            if parameters is None:
                raise ValueError(
                    f'Given filename {filename} does not correspond to the naming scheme')

            # initialize the parameters from the results of the parsing
            self.prefix = parameters.groups()[0]
            self.subject_id = int(parameters.groups()[1])
            self.session_id = int(parameters.groups()[2])
            self.eye = parameters.groups()[3]
            self.x_position = int(parameters.groups()[4])
            self.y_position = int(parameters.groups()[5])
            self.x_size = float(parameters.groups()[6])
            self.y_size = float(parameters.groups()[7])
            self.image_id = int(parameters.groups()[8])
            self.modality = parameters.groups()[9]
            self.postfix = parameters.groups()[10]
            self.extension = parameters.groups()[11]

            self.squared_eccentricity: int = self.x_position ** 2 + self.y_position ** 2
        else:
            self.prefix = None
            self.subject_id = None
            self.session_id = None
            self.eye = None
            self.x_position = None
            self.y_position = None
            self.x_size = None
            self.y_size = None
            self.image_id = None
            self.modality = None
            self.postfix = None
            self.extension = None

            self.squared_eccentricity: int = _DEFAULT_VALUES[_SupportedTypes.POSITIVE_INT]

        _prefix = self.__getattribute__('_prefix')
        if _prefix and _prefix != _default_value(_SupportedTypes.STRING):
            crop_parameters = re.search(
                r'CROP_({})_x({})y({})_'.format(
                    _INT_PATTERN, _INT_PATTERN, _INT_PATTERN
                ), self.prefix
            )

            if crop_parameters is None:
                raise ValueError(
                    f'The prefix {_prefix} does not correspond to the naming scheme')

            self.crop_id = int(crop_parameters.groups()[0])
            self.crop_x_position = int(crop_parameters.groups()[1])
            self.crop_y_position = int(crop_parameters.groups()[2])
        else:
            self.crop_id = None
            self.crop_x_position = None
            self.crop_y_position = None

        self.data = np.empty((0, 0))

    def __str__(self) -> str:

        if floor(self.x_position) == self.x_position:
            x_pos_str = f'{self.x_position :.0f}'
        else:
            x_pos_str = f'{self.x_position}'

        if floor(self.y_position) == self.y_position:
            y_pos_str = f'{self.y_position :.0f}'
        else:
            y_pos_str = f'{self.y_position}'

        try:
            self.prefix = f'CROP_{self.crop_id}_x{self.crop_x_position}y{self.crop_y_position}_'
        except RuntimeError as e:
            if 'is not initialized' in e.args[0]:
                self.prefix = ''
            else:
                raise

        return f'{self.prefix}' \
               f'Subject{self.subject_id}_' \
               f'Session{self.session_id}_' \
               f'{self.eye}_' \
               f'({x_pos_str},{y_pos_str})_' \
               f'{self.x_size}x{self.y_size}_' \
               f'{self.image_id}_' \
               f'{self.modality}' \
               f'{self.postfix if self.postfix is not None else ""}' \
               f'.{self.extension}'

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other) -> bool:
        """
        The relation between images is defined so that they will be sorted from the center to the edges.
        If the distances to the center are equal, then they are first sorted by their X coordinate, then by Y
        coordinate.
        """
        return (self.squared_eccentricity, self.x_position, self.y_position) < \
               (other.squared_eccentricity, other.x_position, other.y_position)

    def read_data(self, directory: str) -> None:
        """
        Stores the data of the represented file in a special field, ready to be accessed.
        Coordinates in the file are stored as (x, y) pairs, but in the object as (y, x).

        WARNING: if the data is cell coordinates, they will be stored as integers!

        :param directory: directory in which the file lies (since there may be several files with the same names in
            different directories, the class itself does not store it)
        :raises ValueError: if it is impossible to read the file, either due to an incompatible format
            or to a wrong path
        """
        raise NotImplementedError

    def get_data(self) -> np.ndarray:
        """
        :return: the data stored in the represented file; coordinates will be converted to (x, y) format
        """
        return self.data

    def set_data(self, data: np.ndarray) -> None:
        self.data = np.copy(np.copy(data))

    def write_data(self, directory: str) -> None:
        raise NotImplementedError

    def is_consistent_with(self, **kwargs) -> bool:
        """
        Checks if the provided parameters are equal to those stored in the object.

        :param kwargs: pairs of parameter_name: parameter_value that need to be checked
        :return: False, if the value of any of the non-None parameters of the object differs from the given one
        :raises ValueError: if any of the given parameters does not exist in the object
        """

        # it is always consistent with an empty set of parameters
        if len(kwargs) == 0:
            return True

        for key in kwargs:
            private_key = '_' + key
            if private_key not in self.__dict__:
                raise ValueError(
                    f'There is no parameter {key} in {type(self).__name__}, you might have made a typo.')

            if self.__dict__[private_key] is not None and self.__dict__[private_key] != kwargs[key]:
                return False
        return True

    def set_parameters_to_compare(self, consider_prefix: bool) -> Dict[str, _ParameterType]:
        parameters_to_compare = {
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'eye': self.eye,
            'x_position': self.x_position,
            'y_position': self.y_position,
            'image_id': self.image_id
        }
        if consider_prefix:
            parameters_to_compare['crop_x_position'] = self.crop_x_position
            parameters_to_compare['crop_y_position'] = self.crop_y_position
        return parameters_to_compare

    def find_corresponding_images(
            self,
            directory: str = None,
            filenames: List[str] = None,
            consider_prefix: bool = False
    ) -> List:
        """
        Searches for the image files with the same parameters. To learn which files are considered images,
        see is_image() method.

        :param consider_prefix: if True, then crop position from prefix will also be taken into account
        :param directory: folder to do the search in
        :return: calculated split file and confocal file (if any of them do not exist, that None instead of it)
        """

        if filenames is None:
            if directory is None:
                raise RuntimeError(
                    'Either a list of filenames or a directory must be specified')
            filenames = os.listdir(directory)

        # print(f'{directory = }, {filenames = }')

        parameters_to_compare = self.set_parameters_to_compare(consider_prefix)
        # print(parameters_to_compare)

        image_files = []

        for filename in filenames:
            try:
                datafile = DataFile(filename)
            except ValueError:
                continue

            if datafile.is_consistent_with(**parameters_to_compare) and datafile.is_image():
                datafile.to_image_file()
                image_files.append(datafile)

        return image_files

    def find_corresponding_coordinates(self, directory: str, consider_prefix: bool = False) -> Optional[Any]:
        """
        Searches for the coordinate files with the same parameters.

        :param directory: folder to do the search in
        :param consider_prefix: if True, then crop position from prefix will also be taken into account
        :return: coordinate file (if does not not exist, that None)
        """

        filenames = os.listdir(directory)
        parameters_to_compare = self.set_parameters_to_compare(consider_prefix)

        for filename in filenames:
            try:
                datafile = DataFile(filename)
            except ValueError:
                continue

            if datafile.is_consistent_with(**parameters_to_compare):
                if datafile.extension in COORDINATES_EXTENSIONS:
                    datafile.to_coordinates_file()
                    return datafile
                else:
                    continue
        return None

    def is_image(self) -> bool:
        """
        :return: True, if the file stores an image, which is determined by its extension
        """
        return self.extension in IMAGE_EXTENSIONS

    def to_image_file(self) -> Any:
        """
        Converts this object to an instance of the ImageFile subclass, to use the corresponding read_data() method.
        """
        self.__class__ = ImageFile
        return self

    def is_coordinates(self) -> bool:
        return self.extension in COORDINATES_EXTENSIONS

    def to_coordinates_file(self) -> Any:
        """
        Converts this object to an instance of the CoordinatesFile subclass, to use the corresponding read_data()
        method.
        """
        self.__class__ = CoordinatesFile
        return self

    def set_parameters(self, **kwargs) -> None:

        # here we have to assign the values via 'if' operators instead of iteration over arguments as we did in
        # self.is_consistent_with(), because we need set the values using each parameter's setter checks

        if 'subject_id' in kwargs:
            self.subject_id = kwargs['subject_id']
        if 'session_id' in kwargs:
            self.session_id = kwargs['session_id']
        if 'eye' in kwargs:
            self.eye = kwargs['eye']
        if 'x_position' in kwargs:
            self.x_position = kwargs['x_position']
        if 'y_position' in kwargs:
            self.y_position = kwargs['y_position']
        if 'x_size' in kwargs:
            self.x_size = kwargs['x_size']
        if 'y_size' in kwargs:
            self.y_size = kwargs['y_size']
        if 'image_id' in kwargs:
            self.image_id = kwargs['image_id']
        if 'modality' in kwargs:
            self.modality = kwargs['modality']
        if 'postfix' in kwargs:
            self.postfix = kwargs['postfix']
        if 'extension' in kwargs:
            self.extension = kwargs['extension']

        if 'crop_id' in kwargs:
            self.crop_id = kwargs['crop_id']
        if 'crop_x_position' in kwargs:
            self.crop_x_position = kwargs['crop_x_position']
        if 'crop_y_position' in kwargs:
            self.crop_y_position = kwargs['crop_y_position']

        if 'crop_id' in kwargs and 'crop_x_position' in kwargs and 'crop_y_position' in kwargs:
            # construct prefix from the given parameters
            self.prefix = f'CROP_{self.crop_id}_x{self.crop_x_position}y{self.crop_y_position}_'
        else:
            # take the given value, if any
            if 'prefix' in kwargs:
                self.prefix = kwargs['prefix']


class ImageFile(DataFile):
    """
    Class to store the images
    """

    def read_data(self, directory: str) -> None:
        """
        Reads the image and put the value of the pixel in the class' data

        :param directory: the folder where the image is
        :raise ValueError: if the image is not in the folder or the extension is
                           wrong
        """
        self.data = cv2.imread(os.path.join(
            directory, self.__str__()), cv2.IMREAD_GRAYSCALE)

        if self.data is None:
            logging.info(os.path.join(directory, self.__str__()))
            raise ValueError(f'{type(self).__name__}: cannot read data, because the extension "{self.extension}" '
                             f'is not supported by OpenCV or the image is not in {directory}.')

    def erase_data(self) -> None:
        """
        Erase the image data from memory
        """
        self.data = None

    def write_data(self, directory: str) -> None:
        """
        Write the image into a directory

        :param directory: the directory to save the image
        :raise ValueError: if the image has not the image format
        """
        if len(self.data.shape) == 2:
            # we write grayscale images as they are
            cv2.imwrite(os.path.join(directory, self.__str__()), self.data)
        elif len(self.data.shape) == 3:
            # opencv writes image with BGR channel order instead of RGB, so we need to revert the image
            cv2.imwrite(os.path.join(directory, self.__str__()),
                        np.flip(self.data, axis=2))
        else:
            raise ValueError('The data is not an image')

    def write_data_for_montage(self, directory: str, ind: int):
        """
        Write data for montage

        :param directory: folder to save the image
        :param ind: the indice of the montage position
        """
        if self.modality == ImageModalities.CO.value:
            method = 'confocal'
        elif self.modality == ImageModalities.CS.value:
            method = 'split_det'
        else:
            method = 'avg'  # dark field
        filename = f'Subject{self.subject_id}_Session{self.session_id}_{self.eye}_{method}_{self.eye}_{self.image_id}' \
                   f'_ref_{ind}_lps_8_lbss_8_ffr_n_50_cropped_5.tif'
        path_to_file = os.path.join(directory, filename)
        if len(self.data.shape) == 2:
            # we write grayscale images as they are
            cv2.imwrite(path_to_file, self.data)
        elif len(self.data.shape) == 3:
            # opencv writes image with BGR channel order instead of RGB, so we need to revert the image
            cv2.imwrite(path_to_file, np.flip(self.data, axis=2))
        else:
            raise ValueError('The data is not an image')


class CoordinatesFile(DataFile):
    """
    Class to store the csv file
    """

    def read_data(self, directory: str) -> None:
        """
        Reads the data from a csv file and store it
        WARNING: inverts the coordinates (if x,y -> y,x)

        :param directory: string with the path to read the file
        :raise ValueError: if the extension is not supported or if the file is
                           not found in the directory
        :raise RuntimeError: if th extension is not csv or txt
        """
        path_to_file = os.path.join(directory, self.__str__())

        if not os.path.exists(path_to_file):
            warnings.warn(
                f'File {path_to_file} does not exist, coordinates will remain empty')
            return

        if self.extension == 'csv':
            with open(path_to_file, 'r') as file:
                reader = csv.reader(file)
                coordinate_list = []
                try:
                    for row in reader:
                        coordinate_list.append(row)
                except UnicodeDecodeError:
                    raise ValueError(f'{type(self).__name__}: cannot read data, because the extension '
                                     f'"{self.extension}" is not supported by the csv reader or the image is not in {directory}.')

                # the list contains floats, so we cannot convert them to integers right away
                coordinates = np.array(
                    coordinate_list, dtype=DEGREE_COORDINATES_DATATYPE)

        elif self.extension == 'txt':
            coordinates = np.loadtxt(path_to_file)

        else:
            raise RuntimeError(
                f'Unsupported file extension "{self.extension}", cannot read data')

        # the coordinates are written in the format (x, y), and we need (y, x)
        if coordinates.size != 0:
            self.data = np.fliplr(np.asarray(
                np.rint(coordinates), dtype=PIXEL_COORDINATES_DATATYPE))
        else:
            self.data = EMPTY_PIXEL_COORDINATES

    def write_data(self, directory: str) -> None:
        """
        Writes the data into a csv file and save it

        :param directory: string with the path to save the file
        :raise RuntimeError: if the extension in not a csv or txt format
        """
        path_to_file = os.path.join(directory, self.__str__())
        # we write coordinate in (x, y) format to be compatible with human-labelled data
        xy_coords = np.fliplr(self.data)

        if self.extension == 'csv':
            with open(path_to_file, 'w', newline='') as file:
                writer = csv.writer(file)
                for row in xy_coords:
                    writer.writerow(row)

        elif self.extension == 'txt':
            np.savetxt(path_to_file, xy_coords)

        else:
            raise RuntimeError(
                f'Unsupported file extension "{self.extension}", cannot write data')


def select_files_with_given_parameters(dir_name: str, **kwargs) -> List[DataFile]:
    """
    Scans the directory for the files that may be converted to a DataFile.
    Among the converted ones select those consistent with the given parameters.

    :param dir_name: directory where to search
    :param kwargs: pairs of parameters in the form <parameter name> = <parameter value>
    :return: list of consistent DataFiles
    """
    datafiles = []
    for filename in os.listdir(dir_name):
        try:
            datafile = DataFile(filename)
        except ValueError:
            # this is not an data file, but some other file that we do not need
            continue

        if datafile.is_consistent_with(**kwargs):
            datafiles.append(datafile)
    return datafiles
