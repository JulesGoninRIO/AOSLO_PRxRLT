#!/usr/bin/env python
"""
Sample script that uses the AutoAOMontaging module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

from __future__ import print_function
import AutoAOMontaging
import matlab
import os
import wmi


def run_auto_ao_montaging(input_dir, loc_file, montage_dir): 
    my_AutoAOMontaging = AutoAOMontaging.initialize()

    # outputdirIn = "C:\\Users\\BardetJ\\Documents\\refactoring\\aoslo_pipeline\\PostProc_Pipe\\Montaging\\AOAutomontaging_master\\montage\\montaged"
    # csvfileIn = "C:\\Users\\BardetJ\\Documents\\refactoring\\aoslo_pipeline\\PostProc_Pipe\\Montaging\\AOAutomontaging_master\\montage\\montaged\\loc.xlsx"
    # inputdirIn = "C:\\Users\\BardetJ\\Documents\\refactoring\\aoslo_pipeline\\PostProc_Pipe\\Montaging\\AOAutomontaging_master\\montage"
    # output_dir = os.path.join(input_dir, montaged_dir)
    loc_file = os.path.join(montage_dir, loc_file)
    my_AutoAOMontaging.AutoAOMontaging(montage_dir, loc_file, input_dir, nargout=0)
    
    # # Fromn https://www.geeksforgeeks.org/how-to-terminate-a-running-process-on-windows-in-python/

    # # This variable ti would be used
    # # as a parity and counter for the
    # # terminating processes
    # ti = 0
    
    # # This variable stores the name
    # # of the process we are terminating
    # # The extension should also be
    # # included in the name
    # matlab = 'MATLAB.exe'
    # excel = 'EXCEL.EXE'

    # # Initializing the wmi object
    # f = wmi.WMI()   
    # # Iterating through all the
    # # running processes
    
    # for process in f.Win32_Process():
        
    #     # Checking whether the process
    #     # name matches our specified name
    #     if process.name == excel:
    
    #         # If the name matches,
    #         # terminate the process   
    #         process.Terminate()
    #         print("Process terminate")
        
    #         # This increment would acknowledge
    #         # about the termination of the
    #         # Processes, and would serve as
    #         # a counter of the number of processes
    #         # terminated under the same name
    #         ti += 1

    #     # Checking whether the process
    #     # name matches our specified name
    #     elif process.name == matlab:
    
    #         # If the name matches,
    #         # terminate the process   
    #         process.Terminate()
    #         print("Process terminate")
        
    #         # This increment would acknowledge
    #         # about the termination of the
    #         # Processes, and would serve as
    #         # a counter of the number of processes
    #         # terminated under the same name
    #         ti += 1
    
    # # True only if the value of
    # # ti didn't get incremented
    # # Therefore implying the
    # # process under the given
    # # name is not found
    # if ti == 0:
    
    #     # An output to inform the
    #     # user about the error
    #     print("Process not found!!!")

    my_AutoAOMontaging.terminate()
