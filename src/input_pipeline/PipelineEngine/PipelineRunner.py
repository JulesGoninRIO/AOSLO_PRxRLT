import configparser
import glob
import logging
import os
import re
import shutil
import sys
import time
from datetime import date
from os.path import isfile, join
from pathlib import Path
from typing import List

from waiting import TimeoutExpired, wait

try:
    from Helpers.data_correction import correct_modality_names
    from Helpers.Exceptions import ListOfStrings, NoModalityError
    from Helpers.os_helpers import empty_folder, remove_file
    from InputData_Pipe.Configs.Parser import Parser
    from InputData_Pipe.ImageProcessing.AOImgProcessing_Manager import \
        AOImgProcessing_Manager
    from InputData_Pipe.PipelineEngine.Date import Date
    from PostProc_Pipe.Helpers.datafile_classes import ImageFile
except ModuleNotFoundError:
    path_to_src = os.path.abspath(os.path.join(
        os.path.abspath(__file__), '..', '..', '..'))
    # sys.path.append(path_to_src)
    from Helpers.data_correction import correct_modality_names
    from Helpers.Exceptions import ListOfStrings, NoModalityError
    from Helpers.os_helpers import empty_folder, remove_file
    from InputData_Pipe.Configs.Parser import Parser
    from InputData_Pipe.ImageProcessing.AOImgProcessing_Manager import \
        AOImgProcessing_Manager
    from InputData_Pipe.PipelineEngine.Date import Date
    from PostProc_Pipe.Helpers.datafile_classes import ImageFile


class PipelineRunner():
    """
    Class that wrap the automated pipeline process that includes loading the Configs
    parameters and the pipeline steps
    """

    def __init__(self, configs_file_name):
        """
        Initialize Config file and the Process Manager class
        """
        Parser.initialize(configs_file_name)
        self.__AOImgProc = AOImgProcessing_Manager()

        # get all the filenames that have been processed
        self.files_processed = []
        self.__modalities_videos = self.__get_video_name()

    def __get_video_name(self) -> List[str]:
        """
        Add string to the modality names to recognize the videos that are inputed

        :return: the modalities with the added video extension
        :rtype: List[str]
        """
        return ["_"+modality+".avi" for modality in Parser.get_modalities()]

    def __getFileFromWaitingRoom(self):
        """
        Continuously look for files in the WatingRoom folder from Configs and once
        it does find AOSLO videos it automatically run the pipeline to process them

        :raises e: _description_
        :return: _description_
        :rtype: _type_
        """
        # get list of files waiting to be processed
        waitingRoom = Parser.get_WaitingRoom_AOImageProc()
        try:
            correct_modality_names(waitingRoom)
        except FileNotFoundError:
            base_dir = Parser.get_AOImageProc_Home()
            waitingRoom = join(base_dir, waitingRoom)
            correct_modality_names(waitingRoom)
        except PermissionError:
            return None

        files = [join(waitingRoom, f) for f in os.listdir(waitingRoom)
                 if os.path.isfile(join(waitingRoom, f)) and not "Thumbs" in f]

        # move oldest file to AOImageProc input folder
        if len(files) > 0:
            logging.info("File found in waiting room")

            for file in files:
                if file not in self.files_processed:
                    self.files_processed.append(file)

            files.sort(key=os.path.getctime)  # order files by time
            modalites_present = []
            files_processed = []

            for file in files:
                if any(modality in file for modality in self.__modalities_videos):
                    modalites_present.append(file)

            AOImageProc_InFolder = Parser.get_AOImageProc_InFolder()

            # make sure we have some modalites present
            if not modalites_present:
                empty_folder(waitingRoom)
                return None

            image_file = ImageFile(Path(modalites_present[0]).name)

            current_subject = image_file.subject_id
            current_session = image_file.session_id
            current_location = (image_file.x_position, image_file.y_position)
            current_number = image_file.image_id
            for file in modalites_present:
                image_file = ImageFile(Path(file).name)
                subject = image_file.subject_id
                session = image_file.session_id
                location = (image_file.x_position, image_file.y_position)
                number = image_file.image_id

                # create the same  temporary name for each file so that the
                # automatic process does not fail to select files
                name = "Subject0_Session000_OD_(0,0)_0x0_0000_" + image_file.modality \
                    + "." + image_file.extension
                logging.info("\nPROCESSING " + str(image_file))

                # move all modality files from waiting room folder to image
                # procesing foler
                original_size = os.path.getsize(file)
                same_image = (current_subject == subject and current_session == session
                              and current_location == location and current_number == number)
                if same_image:
                    files_processed.append(file)
                    success = False
                    moved_file = os.path.join(AOImageProc_InFolder, name)
                    # try until you succeed (the file might still be under creation)
                    while not success:
                        try:
                            logging.info("Moving ...")
                            shutil.move(file, moved_file)
                            success = True
                            logging.info("...success")
                        except PermissionError:
                            logging.info(
                                "Tried to process file which was still being moved: retrying...")
                            time.sleep(3)
                            # retries
                        except FileNotFoundError:
                            moved_file = os.path.join(
                                base_dir, AOImageProc_InFolder, name)

                    # once done moving verify destination file has same size as original file
                    def movingIsDone(): return (original_size == os.path.getsize(moved_file))
                    try:
                        wait(movingIsDone, sleep_seconds=2, timeout_seconds=10,
                             waiting_for="moving file from waiting room to AOImage processing input folder")
                    except TimeoutExpired as e:
                        logging.error(e)
                        # there is no reason why the files should fail to move
                        raise e  # do not recover from this error.
            return files_processed
        else:
            return None  # no new files have been added to waiting room

    def __launchAOImgProcessing(self):
        # launch image processing automation
        self.__AOImgProc.start()
        # wait until image processing is done
        def imageProcessingIsDone(): return (self.__AOImgProc.isAutomation_Done() == True)
        try:
            wait(imageProcessingIsDone, sleep_seconds=5, timeout_seconds=90,
                 waiting_for="AO Image Processing")
        except TimeoutExpired as e:
            # ...keep going even if processing (avi to tif) got stuck
            logging.warning(e)
            # DEBUGGING do not flush: this removes file from waiting room!
            # and causes next retry to fail
            # self.flushPipeline() # clean up before continuing

            # kill AOImgProc GUI before retrying
            self.__AOImgProc.cleanUp()
            logging.info("...retrying")
            return False
        logging.info("...success")
        return True  # if success

    def __copyToMontageServer(self, original_fileNames):
        AOImageProc_OutFolder = Parser.get_AOImageProc_OutFolder()
        output_files_avg = glob.glob(
            str(AOImageProc_OutFolder) + "/*reg_avg.tif")
        output_files_std = glob.glob(
            str(AOImageProc_OutFolder) + "/*reg_std.tif")
        success = True
        if len(output_files_avg) > 0:
            modalities_present = []
            for filename in original_fileNames:
                for modality in Parser.get_modalities():
                    if modality in filename:
                        modalities_present.append(modality)
            for i in range(len(modalities_present)):
                modality_name = [
                    modality for modality in modalities_present if modality in original_fileNames[i]][0]
                AOImageProc_OutFile_avg = [
                    avg for avg in output_files_avg if modality_name in avg]
                AOImageProc_OutFile_std = [
                    std for std in output_files_std if modality_name in std]
                if AOImageProc_OutFile_avg and AOImageProc_OutFile_std:
                    AOImageProc_OutFile_avg = AOImageProc_OutFile_avg[0]
                    AOImageProc_OutFile_std = AOImageProc_OutFile_std[0]
                    Montaging_InFolder = Parser.get_Montaging_InFolder()
                    Montaging_NewFileName = os.path.basename(original_fileNames[i])[
                        :-4]+"_extract_reg_avg.tif"
                    shutil.move(AOImageProc_OutFile_avg, os.path.join(
                        Montaging_InFolder, Montaging_NewFileName))
                    Montaging_NewFileName = os.path.basename(original_fileNames[i])[
                        :-4]+"_extract_reg_std.tif"
                    shutil.move(AOImageProc_OutFile_std, os.path.join(
                        Montaging_InFolder, Montaging_NewFileName))
                else:
                    success = False
        else:
            logging.warning("...I could not find the expected output.")
            logging.warning(
                "...\"Extracted frame count\" was probably 0. Reprocess manually ")
        return success

    def __runPipelineSteps(self):
        """
        Runs the different steps of the automated process once the files are found
        """
        logging.info("Checking waiting room for files")
        continued = True
        # As long as there are still files in the waiting room, the steps will
        # be runned. If no file, the pipeline will keep waiting to get new files.
        waitingroom = Parser.get_WaitingRoom_AOImageProc()
        base_dir = Parser.get_AOImageProc_Home()
        waitingroom_path = join(base_dir, waitingroom)
        while continued:
            # wait until all files have been copied
            past_folder_size =  len(os.listdir(waitingroom_path))
            print(past_folder_size)
            time.sleep(5)
            current_folder_size =  len(os.listdir(waitingroom_path))
            if current_folder_size == past_folder_size and past_folder_size>1:
                files = self.__getFileFromWaitingRoom()
                if files != None:
                    # launch processing (while there are retries left)
                    retries = int(Parser.get_AOImgProc_retriesOnFailure())
                    success = False
                    while retries > 0:
                        logging.info("Launching AOImgProcessing automation...")
                        success = self.__launchAOImgProcessing()
                        if success:
                            # if succesful copy to montage server
                            retries = 0
                            logging.info("Copying to montaging server...")
                            # Sometimes it fails savig std images so if it does so, try again the registration
                            if not self.__copyToMontageServer(files):
                                logging.info("Saving files failed so retries...")
                                retries = retries + 1
                                success = False
                                self.__AOImgProc.cleanUp()
                                AOImageProc_InFolder = Parser.get_AOImageProc_OutFolder() + "/*"
                                for leftover in glob.glob(AOImageProc_InFolder):
                                    os.remove(leftover)
                        else:
                            retries = retries - 1
                    # finished, flush pipeline
                    # print("Finish :", success)
                    if (success):
                        logging.info("...finished")
                    else:
                        logging.warning("...giving up")
                    # print("flush")
                    self.flushPipeline()
                    continued = len(os.listdir(
                        Parser.get_WaitingRoom_AOImageProc())) > 0
                else:
                    # if no file to be processed, sleep
                    sec = int(Parser.get_ActivationTimeInterval_seconds())
                    time.sleep(sec)
            else:
                # files still being uploaded
                sec = int(Parser.get_ActivationTimeInterval_seconds())
                time.sleep(sec)

        # Verify we have processed all files corectly
        failed_to_process = []  # record failed processed files t save as txt file
        files_processed = os.listdir(Parser.get_Montaging_InFolder())
        for file in self.files_processed:
            name = os.path.basename(file)[:-4] + "_extract_reg_avg.tif"
            if name not in files_processed:
                logging.info(
                    f"{file} has not been processed, please try again")
                print(f"{file} has not been processed, please try again")
                failed_to_process.append(name)
            # else :
                # self.files_processed.remove(file)
        with open(os.path.join(Parser.get_Montaging_InFolder(), 'failed_to_process.txt'), 'w') as f:
            f.write('\n'.join(failed_to_process))

    def go(self):
        """
        Continuouly runs the pipeline steps in order by first emptying the directories
        that will be use in the process and then tun the steps of the pipeline
        """
        while True:
            self.flushPipeline()
            self.__runPipelineSteps()

    def flushPipeline(self):
        """
        Clean directories where the files are temporary put when we run through
        the pipeline. It is called before and after the AOImageProcessing steps
        are over.
        """
        logging.info("Flush temporary folders")

        # First sleep so that last files can be finished moving out in output folders
        sec = int(Parser.get_RestAfterRun_seconds())
        time.sleep(sec)

        # Then look for the files and delete them
        AOImageProc_InFolder = Parser.get_AOImageProc_InFolder() + "/*"
        empty_folder(AOImageProc_InFolder[:-1])
        AOImageProc_OutFolder = Parser.get_AOImageProc_OutFolder() + "/*"
        empty_folder(AOImageProc_OutFolder[:-1])
