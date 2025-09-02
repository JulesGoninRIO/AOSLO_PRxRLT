import re
import math
import os
import numpy as np
from typing import Any, Dict, List, Optional, Self, Union
from abc import ABC, abstractmethod
import copy

from src.shared.datafile.validators import (
    _FLOAT_PATTERN, _INT_PATTERN, _LETTER_AND_INT_PATTERN, _LETTER_ONLY_PATTERN,
    _STRING_PATTERN, _ParameterType, _SupportedTypes, _validate_value, _default_value,
    _DEFAULT_VALUES
)
from src.shared.datafile.datafile_constants import COORDINATES_EXTENSIONS, IMAGE_EXTENSIONS

class DataFile(ABC):
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
        :param value_type: id of the type of the expected value for this property
        (field of _SupportedTypes)
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
        :param value_type: id of the type of the expected value for this property
        (field of _SupportedTypes)
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
        return self._property_getter('x_position', _SupportedTypes.INT_OR_FLOAT)

    @x_position.setter
    def x_position(self, value: Union[int, float]) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter('x_position', _SupportedTypes.INT_OR_FLOAT, value)

    @property
    def y_position(self) -> int:
        """
        :raises RuntimeError: if the value has not been initialized before reading
        :raises ValueError: if the given ID does not correspond to any of the supported types
        """
        return self._property_getter('y_position', _SupportedTypes.INT_OR_FLOAT)

    @y_position.setter
    def y_position(self, value: int) -> None:
        """
        :raises RuntimeError: if the given ID does not correspond to any of the supported types
        :raises TypeError: if the value has a wrong type
        """
        self._property_setter('y_position', _SupportedTypes.INT_OR_FLOAT, value)

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
        self._property_setter('crop_id', _SupportedTypes.INT, value)

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

    def __init__(self, filename: str | None = None, datafile=None):
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

        if datafile is not None and filename is not None:
            raise ValueError(
                'Either filename or datafile should be provided, but not both')

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
                parameters = re.search(
                    r'^({})Subject({})_Session({})_({})_\(({}),({})\)_({})x({})_({})_({})({})\.({})$'.format(
                        _STRING_PATTERN, _INT_PATTERN, _INT_PATTERN, _LETTER_ONLY_PATTERN, _FLOAT_PATTERN, _FLOAT_PATTERN,
                        _FLOAT_PATTERN, _FLOAT_PATTERN, _INT_PATTERN, _LETTER_ONLY_PATTERN, _STRING_PATTERN,
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
                self.x_position = float(parameters.groups()[4])
                self.y_position = float(parameters.groups()[5])
                self.x_size = float(parameters.groups()[6])
                self.y_size = float(parameters.groups()[7])
                self.image_id = int(parameters.groups()[8])
                self.modality = parameters.groups()[9]
                self.postfix = parameters.groups()[10]
                self.extension = parameters.groups()[11]

                self.squared_eccentricity: int = self.x_position ** 2 + self.y_position ** 2
                
            else:
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

        if math.floor(self.x_position) == self.x_position:
            x_pos_str = f'{self.x_position :.0f}'
        else:
            x_pos_str = f'{self.x_position}'

        if math.floor(self.y_position) == self.y_position:
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

    def __lt__(self, other: Self) -> bool:
        """
        The relation between images is defined so that they will be sorted from the center to the edges.
        If the distances to the center are equal, then they are first sorted by their X coordinate, then by Y
        coordinate.

        :param other: Another instance of DataFile to compare with.
        :type other: DataFile
        :return: True if this DataFile is less than the other, False otherwise.
        :rtype: bool
        """
        return (
            self.squared_eccentricity, self.x_position, self.y_position
        ) < (other.squared_eccentricity, other.x_position, other.y_position)

    def __eq__(self, other: Self) -> bool:
        """
        Compare this DataFile with another DataFile for equality.

        This method compares the current DataFile instance with another DataFile instance
        to determine if they are equal. It checks each attribute in the __dict__ of both
        instances. If the attribute is a numpy array, it uses numpy's array_equal method
        to compare them. If the attribute is not a numpy array, it uses the standard
        equality operator.

        :param other: Another instance of DataFile to compare with.
        :type other: DataFile
        :return: True if this DataFile is equal to the other, False otherwise.
        :rtype: bool
        """
        for key, value1 in self.__dict__.items():
            value2 = other.__dict__[key]
            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                if value1.size == 0 or value2.size == 0:
                    continue
                elif not np.array_equal(value1, value2):
                    return False
            elif value1 != value2:
                return False

        return True

    def __hash__(self):
        """
        Generate a hash value for the DataFile instance.

        This method generates a hash value for the DataFile instance based on its attributes.
        It tries to include the 'prefix' attribute in the hash calculation. If a RuntimeError
        occurs, it excludes the 'prefix' attribute.

        :return: The hash value of the DataFile instance.
        :rtype: int
        """
        try:
            attributes = (
                self.subject_id, self.session_id, self.eye,
                self.x_position, self.y_position, self.x_size,
                self.y_size, self.image_id, self.modality,
                self.prefix, self.postfix, self.extension
            )
        except RuntimeError:
            attributes = (
                self.subject_id, self.session_id, self.eye,
                self.x_position, self.y_position, self.x_size,
                self.y_size, self.image_id, self.modality,
                self.postfix, self.extension
            )
        return hash(attributes)

    @abstractmethod
    def read_data(self, directory: str) -> None:
        raise NotImplementedError

    def set_data(self, data: np.ndarray) -> None:
        """
        Change the values of the data

        :param data: the array to set the class data to
        :type data: np.ndarray
        """
        self.data = np.copy(data)

    def remove_data(self) -> None:
        """
        Remove the data from the DataFile instance.

        This method sets the 'data' attribute of the DataFile instance to an empty numpy array.

        :return: None
        """
        self.data = np.empty((0, 0))

    @abstractmethod
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
        """
        Compare the parameters of the object with the given ones.

        :param consider_prefix: whether or not we want to include prefix in the comparison
        :type consider_prefix: bool
        :return: The dictionary of the parameters to compare
        :rtype: Dict[str, _ParameterType]
        """
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

    # # TODO: see use cases and refactor if needed -> cannot build a DataFile as it is abstract for now
    # def find_corresponding_files(
    #         self,
    #         directory: str = None,
    #         filenames: List[str] = None,
    #         consider_prefix: bool = False) -> List:
    #     """
    #     Searches for the files with the same parameters and file type.

    #     :param directory: folder to do the search in
    #     :param filenames: list of filenames to search in
    #     :param consider_prefix: if True, then crop position from prefix will also be taken into account
    #     :param file_type: the type of file to search for (e.g., "image", "coordinates")
    #     :return: list of matching files
    #     """

    #     if filenames is None:
    #         if directory is None:
    #             raise RuntimeError(
    #                 'Either a list of filenames or a directory must be specified')
    #         filenames = os.listdir(directory)

    #     parameters_to_compare = self.set_parameters_to_compare(consider_prefix)

    #     matching_files = []

    #     for filename in filenames:
    #         try:
    #             datafile = DataFile(filename)
    #         except ValueError:
    #             continue

    #         if datafile.is_consistent_with(**parameters_to_compare):
    #             if datafile.is_image():
    #                 datafile.to_image_file()
    #             elif datafile.is_coordinates():
    #                 datafile.to_coordinates_file()
    #             matching_files.append(datafile)

    #     return matching_files

    # def is_image(self) -> bool:
    #     """
    #     :return: True, if the file stores an image, which is determined by its extension
    #     """
    #     return self.extension in IMAGE_EXTENSIONS

    # def to_image_file(self) -> Any:
    #     """
    #     Converts this object to an instance of the ImageFile subclass, to use the corresponding read_data() method.
    #     """
    #     self.__class__ = ImageFile
    #     return self

    # def is_coordinates(self) -> bool:
    #     return self.extension in COORDINATES_EXTENSIONS

    # def to_coordinates_file(self) -> Any:
    #     """
    #     Converts this object to an instance of the CoordinatesFile subclass, to use the corresponding read_data()
    #     method.
    #     """
    #     self.__class__ = CoordinatesFile
    #     return self

    def is_patch(self) -> bool:
        """
        :return: True, if the file is a patch
        """
        return self.prefix is not None

    def set_prefix(self, crop_id: int, crop_x_position: int, crop_y_position: int) -> None:
        """
        Sets the prefix of the file according to the given parameters.

        :param crop_id: positive int
        :param crop_x_position: positive int
        :param crop_y_position: positive int
        """
        self.crop_id = crop_id
        self.crop_x_position = crop_x_position
        self.crop_y_position = crop_y_position
        self.prefix = f'CROP_{crop_id}_x{crop_x_position}y{crop_y_position}_'

    def remove_prefix(self):
        """
        Removes the prefix from the filename.
        """
        self.prefix = None
        self.crop_id = None
        self.crop_x_position = None
        self.crop_y_position = None

    def without_prefix(self) -> None:
        """
        Create a copy of the DataFile instance without the prefix.

        This method creates a deep copy of the current DataFile instance and removes the prefix
        from the copied instance.

        :return: A copy of the DataFile instance without the prefix.
        :rtype: DataFile
        """
        obj_copy = copy.deepcopy(self)
        obj_copy.remove_prefix()
        return obj_copy

    def is_same_except_modality(self, other: 'DataFile') -> bool:
        """
        Check if two DataFile instances are the same except for the modality.

        This method compares the current DataFile instance with another DataFile instance
        to determine if they are the same, ignoring differences in the '_modality', '_prefix',
        'data', and '_dark_regions' attributes.

        :param other: Another instance of DataFile to compare with.
        :type other: DataFile
        :return: True if the DataFile instances are the same except for the modality, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, DataFile):
            return False

        for attr in self.__dict__:
            if attr in ['_modality', '_prefix', 'data', '_dark_regions']:
                continue
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def set_parameters(self, **kwargs) -> None:
        """
        Set multiple parameters for the DataFile instance.

        This method sets various attributes of the DataFile instance using the provided keyword arguments.
        It ensures that each parameter is set using its respective setter checks.

        :param kwargs: The keyword arguments containing the parameters to set.
            - subject_id (str): The subject ID.
            - session_id (str): The session ID.
            - eye (str): The eye (e.g., 'left' or 'right').
            - x_position (int): The x position.
            - y_position (int): The y position.
            - x_size (int): The x size.
            - y_size (int): The y size.
            - image_id (str): The image ID.
            - modality (str): The modality.
            - postfix (str): The postfix.
            - extension (str): The file extension.
            - crop_id (str): The crop ID.
            - crop_x_position (int): The crop x position.
            - crop_y_position (int): The crop y position.
            - prefix (str): The prefix (optional, constructed if crop parameters are provided).
        :type kwargs: dict
        :return: None
        """
        # TODO: WARNING WE SHOULD VERIFY THAT ALL PARAMETERS ARE IN KWARGS OTHERWISE
        # CAN LEAD TO INCOMPLETE INITIALIZATION (SEE TESTS), ALSO CAN GIVE INCORRECT
        # KWARGS NAME AND DOES NOTHING WITH IT

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
