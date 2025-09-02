import logging
import os, sys
import subprocess

try:
    from InputData_Pipe.Configs.Parser import Parser
except ModuleNotFoundError:
    path_to_src = os.path.abspath(os.path.join(
        os.path.abspath(__file__), '..', '..', '..'))
    # sys.path.append(path_to_src)
    from InputData_Pipe.Configs.Parser import Parser

import psutil


class AOImgProcessing_Manager():
    """Launch AOImagingProcessing software (from AOSLO avi to image)"""

    def start(self):
        # kill any left over processed before beginning
        self.cleanUp()

        # A (Microsoft) Power Automate Desktop "flows" cannot be invoked from command line
        # I will click on the automation flow link using another UI automation tool (AutoIt)
        UI_automationTool = Parser.get_UI_automationTool()
        UI_automationScript = Parser.get_UI_automationScript()
        powerShell = Parser.get_powerShell()
        # run AutoIT to start Power Automate Desktop
        self.start_process(powerShell, UI_automationTool, UI_automationScript)
        pass

    def start_process(self, powerShell, tool, script):
        call = [powerShell, "START \"" + tool + "\" \"" + script + "\""]
        try:
            subprocess.Popen(call)
        except FileNotFoundError as e:
            # print("Powershell cannot be found")
            logging.error("Powershell cannot be found")
            raise e

    def isAutomation_Done(self):
        # are there output files?
        outputFolder = Parser.get_AOImageProc_OutFolder()
        files = [f for f in os.listdir(outputFolder) if os.path.isfile(
            os.path.join(outputFolder, f))]
        isDone = (len(files) > 0)
        if isDone:
            self.cleanUp()
            return True
        else:
            return False

    def killProcess(self, processName):
        # find PIDs of automation tools
        for p in psutil.process_iter():  # iterate over all active processes
            try:
                if processName in p.name():
                    # kill process (otherwise you fill up memory)
                    powerShell = Parser.get_powerShell()
                    call = [powerShell, "taskkill /F /PID " + str(p.pid)]
                    subprocess.Popen(call)
            except psutil.NoSuchProcess:  # recover from this error...
                continue  # ...processes might have exited in the meanwhile
        pass

    def cleanUp(self):
        self.killProcess("AutoIt3.exe")  # AutoIt
        self.killProcess("AutoIt3_x64.exe")  # AutoIt
        self.killProcess("PAD.Console.Host.exe")  # main Power Automate console
        # Power Automate flow
        self.killProcess("PAD.BridgeToUIAutomation2.exe")
        # Power Automate notification
        self.killProcess("Microsoft.Flow.RPA.Notifier.exe")
        # BMC Image processing GUI
        self.killProcess("AOImageProcessingApp.exe")
