function varargout = AutoAOMontaging(outputFolder_name,postionFile_name,imgfolder_name)
% AUTOAOMONTAGINGGUI MATLAB code for AutoAOMontagingGUI.fig
%      AUTOAOMONTAGINGGUI, by itself, creates a new AUTOAOMONTAGINGGUI or raises the existing
%      singleton*.
%_
%      H = AUTOAOMONTAGINGGUI returns the handle to a new AUTOAOMONTAGINGGUI or the handle to
%      the existing singleton*.
%
%      AUTOAOMONTAGINGGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in AUTOAOMONTAGINGGUI.M with the given input arguments.
%
%      AUTOAOMONTAGINGGUI('Property','Value',...) creates a new AUTOAOMONTAGINGGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before AutoAOMontagingGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to AutoAOMontagingGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help AutoAOMontagingGUI

% Last Modified by GUIDE v2.5 31-Aug-2017 14:05:51

%allocate variables and defaults
% fprintf("MATLAB running");
%outputFolder_name= 'C:\Users\BardetJ\Documents\refactoring\aoslo_pipeline\PostProc_Pipe\Montaging\AOAutomontaging-master\montage\montaged';
combinedFile_names=[];
%postionFile_name= 'C:\Users\BardetJ\Documents\refactoring\aoslo_pipeline\PostProc_Pipe\Montaging\AOAutomontaging-master\montage\montaged\loc.xlsx';
%imgfolder_name= 'C:\Users\BardetJ\Documents\refactoring\aoslo_pipeline\PostProc_Pipe\Montaging\AOAutomontaging-master\montage';
imageFile_names=[];
modalitiesInfo = {'Confocal' 'Confocal';
    'Split Detection' 'CalculatedSplit';
    'Dark Field' 'DarkField'};
inputExt = 1;
device_mode = 'multi_modal';
TransType = 1;
%default to .tif

%add path and setup vl_feat
currentFile = mfilename('fullpath');
[currentFileLoc,name,ext] = fileparts(currentFile); 
genpath(fullfile(currentFileLoc,'SupportFunctions'));
addpath(genpath(fullfile(currentFileLoc,'SupportFunctions')));
% genpath(fullfile(currentFileLoc,'SupportFunctions', 'vlfeat-0.9.20'));
% addpath(genpath(fullfile(currentFileLoc,'SupportFunctions')));
%uigetdir(genpath(fullfile(currentFileLoc,'SupportFunctions')));
% dir(fullfile(fullfile(currentFileLoc,'SupportFunctions'),'*.m'));
%dir(fullfile(currentFileLoc,'SupportFunctions'));
% dir(fullfile(fullfile(fullfile(fullfile(currentFileLoc,'SupportFunctions'), 'vlfeat-0.9.20'), 'toolbox'), '*.m'));
%genpath(fullfile(currentFileLoc,'SupportFunctions')).vl_setup;
% dir(fullfile(currentFileLoc,'SupportFunctions'))/vl_setup.m
vl_setup

% % Check this version of AO Montaging against git.
% fid = fopen(fullfile(getparent(which(mfilename)),'.VERSION'),'r');
% if fid ~= -1
%     thisver = fscanf(fid,'%s');
%     fclose(fid);
%     
%     git_version_check( 'BrainardLab','AOAutomontaging', thisver )
% else
%     warning('Failed to detect .VERSION file, unable to determine if running the newest version.')
% end

Allfiles = dir(fullfile(imgfolder_name,'*.tif'));
Allfiles = {Allfiles.name};

imageFile_names =[];
%Use current identifiers to locate all images


%set(imageList,'String',imageFile_names,...
%    'Value',1)
%guidata(hObject, handles);


% --- Executes on button press in selectPosFile.
% function selectPosFile_Callback(hObject, eventdata, handles)
% hObject    handle to selectPosFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
defaultfolder = imgfolder_name;
%[FileName,PathName] = uigetfile({ '*.csv', 'LUT Files (*.csv)'; '*.xlsx', 'Legacy LUT Format (*.xlsx)' }, ...
%                                'Select LUT file:', defaultfolder);
%[FileName,PathName] = uigetfile({'*.xlsx', 'Legacy LUT Format (*.xlsx)' }, ...
%                                'Select LUT file:', defaultfolder);
%handles.postionFile_name = fullfile(PathName,FileName);
%set(handles.posFileText, 'String', handles.postionFile_name);
%set(handles.posFileText, 'TooltipString', handles.postionFile_name);
%guidata(hObject, handles);


if(~isempty(combinedFile_names) && ~isempty(outputFolder_name))
    index_selected = get(montageList,'Value');
    axes(canvas);
    img = imread(fullfile(outputFolder_name,combinedFile_names{index_selected}));
    imagesc(img(:,:,1)); colormap gray; axis equal; axis off;
end

%New Montage Or Append to Existing?
AppendToExisting=0;
MontageSave=[];
%read filename substrings to search for different modalities


tic
combinedFile_names = AOMosiacAllMultiModal(imgfolder_name, postionFile_name, ...
                                                   outputFolder_name, device_mode, ...
                                                   modalitiesInfo(:,2), TransType, AppendToExisting, ...
                                                   MontageSave, 0);
toc
clear all
