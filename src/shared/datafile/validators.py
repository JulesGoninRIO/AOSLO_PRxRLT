import os
from enum import IntEnum, auto, unique
from pathlib import Path
from typing import Callable, Dict, Union

from src.shared.helpers.exceptions import NotAbsolutePathError, PathDoesNotExistError

def append_message(message: str, file: Path = None) -> str:
    """
    Append the file name to a message

    :param message: the mesasge to send
    :type message: str
    :param file: the image name to add to the message, defaults to None
    :type file: Path, optional
    :return: the updated message
    :rtype: str
    """
    if file:
        return message + f" in the file {file}"
    else:
        return message

def validate_type(value, expected_type, type_name, parameter_name='', file=None):
    """
    Validate that the value given has the proper python type

    :param value: variable to be validated
    :param expected_type: expected type of the value
    :param type_name: name of the expected type
    :param parameter_name: name of the value given, defaults to ''
    :param file: file where the value comes from, defaults to None
    :raises TypeError: if the value is not of the expected type
    """
    if not isinstance(value, expected_type):
        message = f'Expected {parameter_name} to be {type_name}, got {type(value)}'
        raise TypeError(append_message(message, file))

def validate_bool(value: str, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value is of type bool.

    This function validates that the provided value is of type bool. If not, it raises a TypeError.

    :param value: The value to validate.
    :type value: str
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    """
    validate_type(value, bool, 'bool', parameter_name, file)

def validate_int(value: int, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value is of type int.

    This function validates that the provided value is of type int. If not, it raises a TypeError.

    :param value: The value to validate.
    :type value: int
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    """
    validate_type(value, int, 'int', parameter_name, file)

def validate_positive_int(value: int, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value is a positive integer.

    This function validates that the provided value is a positive integer. If not, it raises a ValueError.

    :param value: The value to validate.
    :type value: int
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    :raise ValueError: If the value is not a positive integer.
    """
    validate_int(value, parameter_name, file)
    if value < 0:
        message = f'Expected {parameter_name} to be positive value, got {value}'
        raise ValueError(append_message(message, file))

def validate_string(value: str, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value is of type str.

    This function validates that the provided value is of type str. If not, it raises a TypeError.

    :param value: The value to validate.
    :type value: str
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    """
    validate_type(value, str, 'str', parameter_name, file)

def validate_letter_only_string(value: str, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the string contains only letters.

    This function validates that the provided string contains only letters. If not, it raises a ValueError.

    :param value: The value to validate.
    :type value: str
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    :raise ValueError: If the string contains non-letter characters.
    """
    validate_string(value, parameter_name, file)
    if value and not value.isalpha():
        message = f'Expected {parameter_name} to contain only letters, got "{value}"'
        raise ValueError(append_message(message, file))

def validate_float(value: float, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value is of type float.

    This function validates that the provided value is of type float. If not, it raises a TypeError.

    :param value: The value to validate.
    :type value: float
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    """
    validate_type(value, float, 'float', parameter_name, file)

def validate_int_or_float(value: Union[int, float], parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value is of type int or float.

    This function validates that the provided value is of type int or float. If not, it raises a TypeError.

    :param value: The value to validate.
    :type value: Union[int, float]
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    """
    if not isinstance(value, (int, float)):
        expected_types = 'int or float'
        actual_type = type(value).__name__
        raise TypeError(f"Expected {expected_types} for {parameter_name}, got {actual_type}")

def validate_positive_float(value: float, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value is a positive float.

    This function validates that the provided value is a positive float. If not, it raises a ValueError.

    :param value: The value to validate.
    :type value: float
    :param parameter_name: The name of the parameter being validated.
    :type parameter_name: str, optional
    :param file: The file where the validation is being performed.
    :type file: Path, optional
    :return: None
    :raise ValueError: If the value is not a positive float.
    """
    validate_type(value, float, 'float', parameter_name, file)
    if value < 0.:
        message = f'Expected {parameter_name} to be positive value, got {value}'
        raise ValueError(append_message(message, file))

_INT_PATTERN = r'-?\d+'
_FLOAT_PATTERN = r'-?\d+\.?\d*'
_LETTER_ONLY_PATTERN = r'[a-zA-Z]+'
_LETTER_AND_INT_PATTERN = r'[a-zA-Z0-9]+'
_STRING_PATTERN = r'.*'

_ParameterType = Union[int, float, str]
"""parameters are expected to only have the following types for those validations"""

@unique
class _SupportedTypes(IntEnum):
    """
    Shortcuts for (or IDs of) various datafile parameter types.
    """
    INT = auto()
    POSITIVE_INT = auto()
    FLOAT = auto()
    INT_OR_FLOAT = auto()
    POSITIVE_FLOAT = auto()
    STRING = auto()
    LETTER_ONLY_STRING = auto()
    BOOL = auto()

_VALIDATORS: Dict[int, Callable[[_ParameterType, str], None]] = {
    _SupportedTypes.INT: validate_int,
    _SupportedTypes.POSITIVE_INT: validate_positive_int,
    _SupportedTypes.FLOAT: validate_float,
    _SupportedTypes.INT_OR_FLOAT: validate_int_or_float,
    _SupportedTypes.POSITIVE_FLOAT: validate_positive_float,
    _SupportedTypes.STRING: validate_string,
    _SupportedTypes.LETTER_ONLY_STRING: validate_letter_only_string,
    _SupportedTypes.BOOL: validate_bool
}
"""
functions to validate parameters of supported types; these functions
raise TypeError and ValueError if a provided value of the parameter is not
consistent with its type
"""

def _validate_value(
        value_type: _SupportedTypes,
        value: _ParameterType,
        parameter_name: str = '',
        file: Path = None
    ) -> None:
    """Checks if the value is correct according to its type

    :param value_type: id of one of the available types
    :type value_type: _SupportedTypes
    :param value: value to be validated
    :type value: _ParameterType
    :param parameter_name: name of the parameter being validated, needed to make
    the error message more informative, defaults to ''
    :type parameter_name: str, optional
    :param file: _description_, defaults to None
    :type file: Path, optional
    :raises RuntimeError: if the given ID does not correspond to any of the supported types
    :raises TypeError: if the value has a wrong type
    :raises ValueError: if the value does not belong to a valid range for the
                        desired type
    """
    if value_type not in _VALIDATORS:
        raise RuntimeError('Validator not defined')
    _VALIDATORS[value_type](value, parameter_name, file)

def _validate_path(path: Path, base_dir: Path = None) -> Path:
    """
    Validate that a path exists and if it not an absolute path, will make one from
    the base directory given

    :param path: path to validate
    :type path: Path
    :param base_dir: base path to complete if the path is not absolute, defaults to None
    :type base_dir: Path, optional
    :raises NotAbsolutePathError: if the path is not absolute and we cannot correct
    it with the base directory
    :raises PathDoesNotExistError: if the path does not exists
    :return: the updated verified path
    :rtype: Path
    """
    if not os.path.isabs(path):
        if base_dir:
            path = Path(os.path.join(base_dir, path))
        else:
            raise NotAbsolutePathError(path)
    if not os.path.exists(path):
        raise PathDoesNotExistError(path)
    return path

_DEFAULT_VALUES: Dict[_ParameterType, int] = {
    _SupportedTypes.INT: float('inf'),
    _SupportedTypes.POSITIVE_INT: -1,
    _SupportedTypes.FLOAT: float('inf'),
    _SupportedTypes.POSITIVE_FLOAT: -1.,
    _SupportedTypes.INT_OR_FLOAT: float('inf'), # if set to 0, zeros parsed with _SupportedTypes.INT_OR_FLOAT will trigger DataFile's RuntimeError of _property_getter method
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

