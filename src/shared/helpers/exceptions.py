import logging
from pathlib import Path


class Error(Exception):
    """
    Base class for other exceptions

    :param Exception: the Exception to raise
    :type Exception: Exception
    """

    def __str__(self):
        logging.error(self.message)
        return self.message


class ListOfStrings(Error):
    """
    The list should contain string variables with the "" character around them

    :param Error: the Error to raise
    :type Error: Error
    """

    def __init__(self, file: str = None):
        self.message = "The modality list must be a list of strings. "
        if file:
            self.message += f"Add the modalities in the following Config file: {file}"


class ZeroVector(Error):
    """
    The vector used through numpy will return a nan value

    :param Error: the Error to raise
    :type Error: Error
    """

    def __init__(self, message):
        self.message = f"Unusable array: {message}"


class EmptyVector(Error):
    """
    The vector used is empty

    :param Error: the Error to raise
    :type Error: Error
    """

    def __init__(self, message):
        self.message = f"Empty array: {message}"


class NoModalityError(Error):
    """
    The modality list is empty

    :param Exception: the Exception to raise
    :type Exception: Exception
    """

    def __init__(self, file: str = None):
        self.message = "The modality list is empty. "
        if file:
            self.message += f"Add the modalities in the following Config file: {file}"


class NoSectionError(Error):
    """
    There is not the demanded section in the config file

    :param Error: the Error to raise
    :type Error: Error
    """

    def __init__(self, section: str, file: str = None):
        self.message = f"No section {section}. "
        if file:
            self.message += f"Add the section in the following Config file: {file}"


class NoOptionError(Error):
    """
    There is not the demanded option in the config file

    :param Error: the Error to raise
    :type Error: Error
    """

    def __init__(self, option: str, file: str = None):
        self.message = f"No option {option}. "
        if file:
            self.message += f"Add the option in the following Config file: {file}"


class NotAbsolutePathError(Error):
    """
    There is not the demanded option in the config file

    :param Error: the Error to raise
    :type Error: Error
    """

    def __init__(self, file: Path):
        self.message = f"The {file} is not an absolute path."


class PathDoesNotExistError(Error):
    """
    The Path does not exist

    :param Error: the Error to raise
    :type Error: Error
    """

    def __init__(self, file: Path):
        self.message = f"The {file} does not exists"
