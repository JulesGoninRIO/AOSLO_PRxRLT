import configparser
import os

from Helpers.Exceptions import (NoOptionError, NoSectionError,
                                NotAbsolutePathError, PathDoesNotExistError)
from Helpers.read_config import add_config_option, read_config_line, read_list
from InputData_Pipe.Configs.constants import (ACTIVATIONTIMEINTERVAL_SECONDS,
                                              AOIMGPROC_RETRIESONFAILURE,
                                              RESTAFTERRUN_SECONDS)
from PostProc_Pipe.Helpers.validators import (_ParameterType, _SupportedTypes,
                                              _validate_path, _validate_value)


class Parser():
    """
    Gets all the informations from config file with optional and needed sections
    and options
    """

    @staticmethod
    def initialize(configs_file):
        """
        Static method to initialize the mandatory options
        """
        # The configs file should lie in the same folder as this script
        __configs_file = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                                      '..', configs_file))
        Parser.configs = configparser.RawConfigParser()
        Parser.configs.read(__configs_file)

        # If the configparser is empty it means the Config file has not been found
        if len(Parser.configs._sections) == 0:
            raise FileNotFoundError(configs_file)
        # add path of the config file for error handling
        Parser.__configs = add_config_option(Parser.configs, configs_file)

        # Get the mandatory options from the config file
        Parser.__powerShell = read_config_line(
            Parser.configs, "os", "__powerShell")
        _validate_value(_SupportedTypes.STRING,
                        Parser.__powerShell, configs_file)
        Parser.__WaitingRoom_AOImageProc = read_config_line(Parser.configs,
                                                            "AOImgProc", "__WaitingRoom_AOImageProc")
        _validate_value(_SupportedTypes.STRING, Parser.__WaitingRoom_AOImageProc,
                        "__WaitingRoom_AOImageProc", configs_file)
        Parser.__AOImageProc_InFolder = read_config_line(Parser.configs,
                                                         "AOImgProc", "__AOImageProc_InFolder")
        _validate_value(_SupportedTypes.STRING, Parser.__AOImageProc_InFolder,
                        "__AOImageProc_InFolder", configs_file)
        Parser.__AOImageProc_OutFolder = read_config_line(Parser.configs,
                                                          "AOImgProc", "__AOImageProc_OutFolder")
        _validate_value(_SupportedTypes.STRING, Parser.__AOImageProc_OutFolder,
                        "__AOImageProc_OutFolder", configs_file)
        Parser.__Montaging_InFolder = read_config_line(Parser.configs,
                                                       "Montaging", "__Montaging_InFolder")
        _validate_value(_SupportedTypes.STRING, Parser.__Montaging_InFolder,
                        "__Montaging_InFolder", configs_file)
        Parser.__modalities = read_list(
            Parser.configs, "AOImgProc", "__modalities")
        Parser.__UI_automationTool = read_config_line(Parser.configs,
                                                       "AOImgProc", "__UI_automationTool")
        Parser.__UI_automationScript = read_config_line(Parser.configs,
                                                       "AOImgProc", "__UI_automationScript")

        # Get the optional options from the Config and if they are not written,
        # will load the default constant parameters
        try:
            Parser.__ActivationTimeInterval_seconds = eval(read_config_line(
                Parser.configs, "Pipeline", "__ActivationTimeInterval_seconds"))
        except (NoSectionError, NoOptionError):
            Parser.__ActivationTimeInterval_seconds = ACTIVATIONTIMEINTERVAL_SECONDS
        _validate_value(_SupportedTypes.POSITIVE_INT,
                        Parser.__ActivationTimeInterval_seconds, "__ActivationTimeInterval_seconds",
                        configs_file)
        try:
            Parser.__RestAfterRun_seconds = eval(read_config_line(Parser.configs,
                                                                  "Pipeline", "__RestAfterRun_seconds"))
        except (NoSectionError, NoOptionError):
            Parser.__RestAfterRun_seconds = RESTAFTERRUN_SECONDS
        _validate_value(_SupportedTypes.POSITIVE_INT, Parser.__RestAfterRun_seconds,
                        "__RestAfterRun_seconds", configs_file)
        try:
            Parser.__AOImgProc_retriesOnFailure = eval(read_config_line(Parser.configs,
                                                                        "AOImgProc", "__AOImgProc_retriesOnFailure"))
        except NoOptionError:
            Parser.__AOImgProc_retriesOnFailure = AOIMGPROC_RETRIESONFAILURE
        _validate_value(_SupportedTypes.POSITIVE_INT, Parser.__AOImgProc_retriesOnFailure,
                        "__AOImgProc_retriesOnFailure", configs_file)
        try:
            Parser.__AOImageProc_Home = read_config_line(Parser.configs, "AOImgProc",
                                                         "__AOImageProc_Home")
        except NoOptionError:
            path_to_project_root = os.path.abspath(os.path.join(
                os.path.abspath(__file__), '..', '..', '..', '..'))
            Parser.__AOImageProc_Home = path_to_project_root
        _validate_value(_SupportedTypes.STRING, Parser.__AOImageProc_Home,
                        "__AOImageProc_Home", configs_file)

        # Make sure we have absolute path for all the directory names
        Parser.__powerShell = _validate_path(Parser.__powerShell,
                                             Parser.__AOImageProc_Home)
        Parser.__WaitingRoom_AOImageProc = _validate_path(Parser.__WaitingRoom_AOImageProc,
                                                          Parser.__AOImageProc_Home)
        Parser.__AOImageProc_InFolder = _validate_path(Parser.__AOImageProc_InFolder,
                                                       Parser.__AOImageProc_Home)
        Parser.__AOImageProc_OutFolder = _validate_path(Parser.__AOImageProc_OutFolder,
                                                        Parser.__AOImageProc_Home)
        Parser.__Montaging_InFolder = _validate_path(Parser.__Montaging_InFolder,
                                                     Parser.__AOImageProc_Home)

    @staticmethod
    def get_powerShell():
        """
        Static method to get the PowerShell path
        """
        return Parser.__powerShell

    @staticmethod
    def get_WaitingRoom_AOImageProc():
        """
        Static method to get the Waiting room directory
        """
        return Parser.__WaitingRoom_AOImageProc

    @staticmethod
    def get_AOImageProc_InFolder():
        """
        Static method to get the directory where we will place images at first
        """
        return Parser.__AOImageProc_InFolder

    @staticmethod
    def get_AOImageProc_OutFolder():
        """
        Static method to get the directory where the image will end up once processed
        """
        return Parser.__AOImageProc_OutFolder

    @staticmethod
    def get_Montaging_InFolder():
        """
        Static method to get the directory to pace images for the montage
        """
        return Parser.__Montaging_InFolder

    @staticmethod
    def get_modalities():
        """
        Static method to get the modalities to process
        """
        return Parser.__modalities

    @staticmethod
    def get_ActivationTimeInterval_seconds():
        """
        Static method to get the interval between activation
        """
        return Parser.__ActivationTimeInterval_seconds

    @staticmethod
    def get_RestAfterRun_seconds():
        """
        Static method to get the time to rest after a process
        """
        return Parser.__RestAfterRun_seconds

    @staticmethod
    def get_AOImgProc_retriesOnFailure():
        """
        Static method to get the number of retries when the pipeline fails
        """
        return Parser.__AOImgProc_retriesOnFailure

    @staticmethod
    def get_AOImageProc_Home():
        """
        Static method to get the base directory
        """
        return Parser.__AOImageProc_Home

    @staticmethod
    def get_UI_automationTool():
        """
        Static method to get the base directory
        """
        return Parser.__UI_automationTool

    @staticmethod
    def get_UI_automationScript():
        """
        Static method to get the base directory
        """
        return Parser.__UI_automationScript
