import os
import numpy as np
from typing import Union, Dict, Callable, Optional, Any, List
from enum import unique, auto, IntEnum, Enum
import sys
from pathlib import Path
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

def validate_bool(value: str, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value given has the proper python bool type

    :param value: variable to be validated
    :type value: str
    :param parameter_name: name of the value given, defaults to ''
    :type parameter_name: str, optional
    :param file: file where the value comes from, defaults to None
    :type file: Path, optional
    :raises TypeError: if the value is not a boolean
    """
    if not isinstance(value, bool):
        message = f'Expected {parameter_name} to be bool, got {type(value)}'
        raise TypeError(append_message(message, file))

def validate_int(value: int, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value given has the proper python int type

    :param value: variable to be validated
    :type value: int
    :param parameter_name: name of the value given, defaults to ''
    :type parameter_name: str, optional
    :param file: file where the value comes from, defaults to None
    :type file: Path, optional
    :raises TypeError: if the value is not integer
    """
    if not isinstance(value, int) and not np.issubdtype(type(value), np.integer):
        message = f'Expected {parameter_name} to be int, got {type(value)}'
        raise TypeError(append_message(message, file))


def validate_positive_int(value: int, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value given has the proper python positive int type

    :param value: variable to be validated
    :type value: int
    :param parameter_name: name of the value given, defaults to ''
    :type parameter_name: str, optional
    :param file: file where the value comes from, defaults to None
    :type file: Path, optional
    :raises TypeError: if the value is not integer
    :raises ValueError: if the value is negative
    """
    validate_int(value, parameter_name)
    if value < 0:
        message = f'Expected {parameter_name} to be positive value, got {value}'
        raise ValueError(append_message(message, file))


def validate_string(value: str, parameter_name: str = '', file: Path = None):
    """
    Validate that the value given has the proper python string type

    :param value: variable to be validated
    :type value: str
    :param parameter_name: name of the value given, defaults to ''
    :type parameter_name: str, optional
    :param file: file where the value comes from, defaults to None
    :type file: Path, optional
    :raises TypeError: if the value is not a string
    """
    if not isinstance(value, str):
        message = f'Expected {parameter_name} to be str, got {type(value)}'
        raise TypeError(append_message(message, file))


def validate_letter_only_string(value: str, parameter_name: str = '', file: Path = None):
    """
    Validate that the value given has the proper python string type with only letters

    :param value: variable to be validated
    :type value: str
    :param parameter_name: name of the value given, defaults to ''
    :type parameter_name: str, optional
    :param file: file where the value comes from, defaults to None
    :type file: Path, optional
    :raises TypeError: if the value is not a string
    :raises ValueError: if the string is not empty and contains other symbols than letters
    """
    validate_string(value, parameter_name)
    if value and not value.isalpha():
        message = f'Expected {parameter_name} to contain only letters, got "{value}"'
        raise ValueError(append_message(message, file))


def validate_float(value: float, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value given has the proper python float type

    :param value: variable to be validated
    :type value: float
    :param parameter_name: name of the value given, defaults to ''
    :type parameter_name: str, optional
    :param file: file where the value comes from, defaults to None
    :type file: Path, optional
    :raises TypeError: if the value is not float
    """
    if not isinstance(value, float) and not np.issubdtype(type(value, np.floating)):
        message = f'Expected {parameter_name} to be float, got {type(value)}'
        raise TypeError(append_message(message, file))


def validate_positive_float(value: float, parameter_name: str = '', file: Path = None) -> None:
    """
    Validate that the value given has the proper python float type with positive values

    :param value: variable to be validated
    :type value: float
    :param parameter_name: name of the value given, defaults to ''
    :type parameter_name: str, optional
    :param file: file where the value comes from, defaults to None
    :type file: Path, optional
    :raises TypeError: if the value is not float
    :raises ValueError: if the value is a negative float
    """
    validate_float(value, parameter_name)
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
    POSITIVE_FLOAT = auto()
    STRING = auto()
    LETTER_ONLY_STRING = auto()
    BOOL = auto()

_VALIDATORS: Dict[int, Callable[[_ParameterType, str], None]] = {
    _SupportedTypes.INT: validate_int,
    _SupportedTypes.POSITIVE_INT: validate_positive_int,
    _SupportedTypes.FLOAT: validate_float,
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
    :raises RuntimeError: if the given ID does not correspond to any of the
                          supported types
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