import configparser
import os
import sys
from typing import List

from src.shared.helpers.exceptions import (ListOfStrings, NoModalityError, NoOptionError, NoSectionError)

SECTION_NAME = "Name"
OPTION_NAME = "__config_name"


def read_config_line(configs_file: configparser, section: str, option: str) -> str:
    """
    Read a line of the config file

    :param configs_file: the config file as a configparser object
    :type configs_file: configparser
    :param section: the section to find the required value
    :type section: str
    :param option: the option to find the required value
    :type option: str
    :raises NoSectionError: if the section does not exist
    :raises NoOptionError: if the option does not exist
    :return: the value of the configuration line needed
    :rtype: str
    """

    try:
        return configs_file.get(section, option)
    except configparser.NoSectionError:
        config_name = read_name(configs_file)
        raise NoSectionError(section, config_name)
    except configparser.NoOptionError:
        config_name = read_name(configs_file)
        raise NoOptionError(option, config_name)


def add_config_option(configs_file: configparser,
                      value: str,
                      section: str = SECTION_NAME,
                      option: str = OPTION_NAME) -> configparser:
    """
    Add a config option to the config file

    :param configs_file: the config file as a configparser object
    :type configs_file: configparser
    :param value: the value to put in the config file
    :type value: str
    :param section: section where to put the value in the config, defaults to SECTION_NAME
    :type section: str, optional
    :param option: option where to put the value in the config, defaults to OPTION_NAME
    :type option: str, optional
    :return: the config file as a configparser object
    :rtype: configparser
    """

    try:
        configs_file.set(section, option, value)
    except configparser.NoSectionError:
        configs_file.add_section(section)
        configs_file.set(section, option, value)
    return configs_file


def read_name(configs_file: configparser) -> str:
    """
    Read the name of the file if it exists

    :param configs_file: the config file as a configparser object
    :type configs_file: configparser
    :return: the name of the config file if exists else None
    :rtype: str
    """

    try:
        config_name = configs_file.get(SECTION_NAME, OPTION_NAME)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None
    return config_name


def read_list(configs_file: configparser, section: str, option: str) -> List:
    """
    Read the config file where the value needs to be a non-empty list of strings

    :param configs_file: the config file as a configparser object
    :type configs_file: configparser
    :param section: the section to find the required list
    :type section: str
    :param option: the option to find the required list
    :type option: str
    :raises ListOfStrings: if one or many elements of the list are not strings
    :raises NoModalityError: if the list is emptys
    :return: the list found in the config file
    :rtype: List
    """

    try:
        list = eval(read_config_line(configs_file, section, option))
    except SyntaxError:
        # The elements of the list are not strings
        raise ListOfStrings(configs_file)
    except NameError:
        # One or many (but not all) element of the list is not a string
        raise ListOfStrings(configs_file)

    if len(list) < 1:
        raise NoModalityError()

    return list
