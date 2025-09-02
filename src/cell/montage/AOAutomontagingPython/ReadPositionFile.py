import os
import numpy as np
import pandas as pd
import re
from typing import Tuple

class ReadPositionFile:
    """
    Reads and processes position data from a given file.

    The file can be either a CSV or an Excel file. The class stores information about image filenames,
    pixel scale, location coordinates, number of columns in the file, eye side, and any errors that occur
    during processing.

    """
    def __init__(self, imageDir: str, inData: list, posFileLoc: str, MN: int, N: int):
        """
        :param imageDir: The directory where the images are located
        :type imageDir: str
        :param inData: The input data
        :type inData: list
        :param posFileLoc: The location of the position file
        :type posFileLoc: str
        :param MN: The number of modalities
        :type MN: int
        :param N: The number of images
        :type N: int
        """
        self.imageDir = imageDir
        self.inData = inData
        self.inData = inData
        self.posFileLoc = posFileLoc
        self.MN = MN
        self.N = N
        self.imageFilename = [[None]*N for _ in range(MN)]  # stores all filenames
        self.pixelScale = np.full(N, np.nan)  # Stores the relative size of our images
        self.LocXY = np.full((2, N), np.nan)  # set NaN to start
        self.NC = -1  # number of columns in the file
        self.eyeSide = ['OS']*N
        self.errorFlag = None

    def readPositionFile(self) -> Tuple[list, list, list, list, int, str]:
        """
        Reads the position file and processes it based on its file type.

        :return: The image filenames, eye sides, pixel scales, location coordinates,
                 number of columns, and any errors
        :rtype: tuple
        """
        if self.posFileLoc.endswith('.csv'):
            self.process_csv_file()
        else:
            self.process_excel_file()
        return self.imageFilename, self.eyeSide, self.pixelScale, self.LocXY, \
            self.NC, self.errorFlag


    def process_csv_file(self):
        """
        Processes a CSV file to extract image information.
        """
        lut = pd.read_csv(self.posFileLoc,
                          names=['img_name', 'X', 'Y', 'range'],
                          sep=';',
                          index_col=False).drop_duplicates()
        lut.reset_index(drop=True, inplace=True)
        version = lut.columns[0]

        if version == 'v0_1':
            lut.iloc[:, 0] = lut.iloc[:, 0].apply(lambda x: f"_{x:04.0f}_")

        self.imageFilename = [[os.path.join(self.imageDir, self.inData[m][n]) \
            for n in range(self.N)] for m in range(self.MN)]
        numbers = [re.search(r'_\d+_', self.inData[0][n]).group().replace('_', '') \
            for n in range(self.N)]
        indices = [lut.iloc[:, 0] == int(number) for number in numbers]
        self.ID = [lut.loc[indice, 'img_name'].values[0] for indice in indices]
        self.LocXY = np.array([[lut.loc[indice, 'X'].values[0], -lut.loc[indice, 'Y'].values[0]] \
            for indice in indices]).T
        self.pixelScale = [lut.loc[indice, 'range'].values[0] for indice in indices]
        filenames = [os.path.basename(self.imageFilename[0][n]) for n in range(self.N)]
        self.eyeSide = ['OD' if '_OD_' in filename else 'OS' if '_OS_' in filename else '' \
            for filename in filenames]

    def process_excel_file(self):
        """
        Processes an Excel file to extract image information.
        """
        # String match that is expected to show up in the filename of each image.
        # E.g. '_0018_ref_7_'
        matchexp = r'_\d\d\d\d_ref_\d'
        # load position info from excel spreadsheet
        C = pd.read_excel(self.posFileLoc).values
        C = C.astype(str)
        NC = C.shape[1]
        # First look for key info like which eye
        for i in range(C.shape[0]):
            if C[i, 0].lower() == 'eye':
                eyeSide = C[i, 1]

        # Then convert back to a number, before adding the trappings of our file ids.
        C[:, 0] = [f"_{int(float(x)):04.0f}_" for x in C[:, 0]]

        # verify that the image id's line up for all modalities
        for n in range(self.N):
            # build filename structure and check to make sure all modalities are present for all ids
            self.imageFilename[0][n] = os.path.join(self.imageDir, self.inData[0][n])
            ImageID_m1 = re.findall(matchexp, self.inData[0][n])
            for m in range(1, self.MN):
                self.imageFilename[m][n] = os.path.join(self.imageDir, self.inData[m][n])
                ImageID_mf = re.findall(matchexp, self.inData[m][n])
                if ImageID_m1 != ImageID_mf:  # check for errors
                    self.errorFlag = f"Error: Mismatch detected."
                    self.errorFlag+=f"Every image number must have the same number of modalities."
                    self.errorFlag+=f"Check image {ImageID_m1}"
                    return

            # match with info from excel
            for i in range(C.shape[0]):
                if C[i, 0] in self.inData[0][n]:
                    if NC >= 4:
                        scale = float(C[i, 3].strip())
                        if not np.isnan(scale) and scale > 0:
                            self.pixelScale[n] = scale

                    # first try looking at coordinate grid
                    if NC >= 3:
                        Loc = C[i, 2].split(',')
                        if len(Loc) == 2:
                            self.LocXY[0, n] = float(Loc[0].strip())
                            self.LocXY[1, n] = float(Loc[1].strip())

                    if NC >= 2:
                        # coordinate grind c
                        if np.isnan(self.LocXY[0, n]) or np.isnan(self.LocXY[1, n]):
                            self.LocXY[:, n] = parseShorthandLoc(C[i, 1], eyeSide)

                    break

            # If we can't find
            if np.isnan(self.LocXY[0, n]) or np.isnan(self.LocXY[1, n]) or \
                np.isnan(self.pixelScale[n]):
                print(f"Warning:")
                print(f"Location information missing or invalid for image cluster: {self.imageFilename[0][n]}")
                for m in range(self.MN):
                    self.imageFilename[m][n] = None

        self.pixelScale = np.max(self.pixelScale) / self.pixelScale

def parseShorthandLoc(inputString: str, eyeSide: str) -> np.ndarray:
    """
    Parses shorthand strings to determine coordinate locations.

    :param inputString: The string to parse
    :type inputString: str
    :param eyeSide: The side of the eye being observed (OD or OS)
    :type eyeSide: str
    :return: The location parsed from the string
    :rtype: numpy.ndarray
    """

    inputString = inputString.upper()  # make input all caps just in case
    LocXY = np.array([np.nan, np.nan])

    # Flip x coordinate if eye is 'OD'
    EyeFlip = 1
    if eyeSide.upper() == 'OD':
        EyeFlip = -1

    # check foveal cases first
    shorthand_dict = {'TL': [1, 1], 'TM': [0, 1], 'TR': [-1, 1],
                      'ML': [1, 0], 'C': [0, 0], 'MR': [-1, 0],
                      'BL': [1, -1], 'BM': [0, -1], 'BR': [-1, -1]}

    if inputString in shorthand_dict:
        LocXY = np.array(shorthand_dict[inputString])
    else:
        # if not foveal case then parse
        lettersLoc = [i for i, c in enumerate(inputString) if c.isalpha()]
        LN = len(lettersLoc)
        N = len(inputString)
        # check that format fits 1 or 2 letters each followed by numbers
        if LN > 2 or LN < 1 or N < 2 or (LN == 2 and N < 4):
            return LocXY
        else:  # if formats fits then we initialize with zeros
            LocXY = np.array([0, 0])

        if lettersLoc[0] == 0:  # Letter in front format
            for i in range(LN):
                if LN == 2 and i == 0:
                    lastNum = lettersLoc[1] - 1
                else:
                    lastNum = N
                if inputString[lettersLoc[i]] == 'T':
                    LocXY[0] = EyeFlip * float(inputString[lettersLoc[i] + 1:lastNum])
                elif inputString[lettersLoc[i]] == 'N':
                    LocXY[0] = -EyeFlip * float(inputString[lettersLoc[i] + 1:lastNum])
                elif inputString[lettersLoc[i]] == 'S':
                    LocXY[1] = float(inputString[lettersLoc[i] + 1:lastNum])
                elif inputString[lettersLoc[i]] == 'I':
                    LocXY[1] = -float(inputString[lettersLoc[i] + 1:lastNum])
        elif lettersLoc[LN - 1] == N - 1:  # Letter Behind Format
            for i in range(LN):
                if LN == 2 and i == 1:
                    firstNum = lettersLoc[0] + 1
                else:
                    firstNum = 0
                if inputString[lettersLoc[i]] == 'T':
                    LocXY[0] = EyeFlip * float(inputString[firstNum:lettersLoc[i]])
                elif inputString[lettersLoc[i]] == 'N':
                    LocXY[0] = -EyeFlip * float(inputString[firstNum:lettersLoc[i]])
                elif inputString[lettersLoc[i]] == 'S':
                    LocXY[1] = float(inputString[firstNum:lettersLoc[i]])
                elif inputString[lettersLoc[i]] == 'I':
                    LocXY[1] = -float(inputString[firstNum:lettersLoc[i]])

    return LocXY
