import numpy as np
import os
from scipy.io import loadmat
from skimage import io
import time
import os
import scipy.io
import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize
# from vlfeat import vl_sift
# from gridFeatures import gridFeatures
# from .filterSiftFeaturesByROI import filter_sift_features_by_roi
#import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize
from .ReadPositionFile import ReadPositionFile
from .calculateFeatures import calculateFeatures
import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize
from scipy.linalg import pinv
import os
from skimage import io, img_as_float
from skimage.transform import resize, AffineTransform, warp
from scipy.linalg import pinv
import numpy as np
import cv2
from typing import Tuple, List
import sys
from .helpers import initialize_bounding_box, pad_image, sift_mosaic_fast_MultiModal


from src.shared.computer_vision.image import shift_image
from src.shared.numpy.arrays import rotate_image

class AOMosaicAllMultiModal:
    def __init__(self, imageDir, posFileLoc, outputDir, ModalitiesSrchStrings):
        """
        Main AO Montaging Function that creates a full montage from an input
        directory with images and nominal coordinate location
        """
        # Algorithm Parameters

        # Threshold for number of SIFT matches needed before accepting the transformation
        self.NumOkMatchesThresh = 10
        self.imageDir = imageDir
        self.posFileLoc = posFileLoc
        self.outputDir = outputDir
        self.ModalitiesSrchStrings = ModalitiesSrchStrings
        self.organizeDataByModality()
        self.N = len(self.inData[0])

    def organizeDataByModality(self):
        """
        Organizes data by modality.

        :return: A tuple containing the organized data,
            the number of non-empty modalities, and an error flag (if any)
        :rtype: tuple
        """
        Allfiles = [f for f in os.listdir(self.imageDir) if os.path.splitext(f)[1] == '.tif']

        self.MN = len(self.ModalitiesSrchStrings)
        self.inData = []
        self.errorFlag = None

        counter = 0
        for m in range(self.MN):
            if self.ModalitiesSrchStrings[m]:  # check it's not empty
                # search for image of this modality
                imagesFound = sorted([f for f in Allfiles if self.ModalitiesSrchStrings[m] in f])
                if imagesFound:
                    if not self.inData or len(imagesFound) == len(self.inData[0]):
                        self.inData.append(imagesFound)
                        counter += 1
                    else:
                        self.errorFlag = 'Error: Mismatch detected.'
                        self.errorFlag += 'Dataset must have the same number of modalities.'
                        return

        self.MN = counter  # only using nonempty identifiers

        for n in range(len(self.inData[0])):
            compstrs = [self.inData[m][n].replace(self.ModalitiesSrchStrings[m], '') \
                for m in range(self.MN) if self.ModalitiesSrchStrings[m]]

            if len(set(compstrs)) > 1:
                self.errorFlag = f'Error: All image pairs/tuples must have all available modalities.'
                self.errorFlag+='Check image {self.inData[0][n]}.'
                return

            compstrs = np.array(list(map(list, compstrs)))
            if not np.all(compstrs[0, :] == compstrs):
                self.errorFlag = f'Error: All image pairs/tuples must have all available modalities.'
                self.errorFlag+='Check image {self.inData[0][n]}.'
                return

    def send_failure(self) -> list:
        """
        Handle a failure in the processing of the image.

        :return: An empty list, indicating that no output was generated due to the failure.
        :rtype: list
        """
        print(self.errorFlag)
        outNameList = []
        return outNameList

    def run(self):

        # catch errors
        if self.errorFlag:
            self.send_failure()

        self.initialize_variables()

        # read position file
        PosFile = ReadPositionFile(self.imageDir, self.inData, str(self.outputDir / self.posFileLoc), self.MN, self.N)
        #TODO: for now eyeSide is useless, will need to change that
        self.imageFilename, eyeSide, self.pixelScale, self.LocXY, self.NC, \
            self.errorFlag = PosFile.readPositionFile()
        # catch errors
        if self.errorFlag:
            self.send_failure()

        # sort using LocXY
        self.sortUsingLocXY()

        f_all, d_all = calculateFeatures(self.imageFilename, self.pixelScale, self.MN, self.N)
        # f_all = result[0]
        # d_all = result[1]

        # # import pickle
        def keypoints_to_dict(keypoints):
            if keypoints is None:
                return None
            else:
                return [{'pt': kp.pt, 'size': kp.size, 'angle': kp.angle,
                         'response': kp.response, 'octave': kp.octave,
                         'class_id': kp.class_id} for kp in keypoints]
        def dict_to_keypoints(keypoints_dict):
            if keypoints_dict is None:
                return None
            else:
                return [cv2.KeyPoint(x=kp['pt'][0], y=kp['pt'][1], _size=kp['size'],
                                     _angle=kp['angle'], _response=kp['response'],
                                     _octave=kp['octave'],
                                     _class_id=kp['class_id']) for kp in keypoints_dict]

        # f_all_serializable = [[keypoints_to_dict(kp) for kp in sublist] for sublist in f_all]
        import pickle
        # with open('f_all_big_dddd_105.pkl', 'wb') as f:
        #     pickle.dump(f_all_serializable, f)
        # with open('d_all_big_dddd_105.pkl', 'wb') as f:
        #     pickle.dump(d_all, f)

        # from scipy.io import loadmat
        # f_all = loadmat(r'C:\Users\BardetJ\Downloads\aoslo_pipeline-master\src\PostProc_Pipe\Montaging\AOAutomontaging_master\f_all.mat')
        # d_all = loadmat(r'C:\Users\BardetJ\Downloads\aoslo_pipeline-master\src\PostProc_Pipe\Montaging\AOAutomontaging_master\d_all.mat')

        # f_all = [[np.transpose(arr) for arr in sublist] for sublist in f_all['f_all']]
        # d_all = [[np.transpose(arr) for arr in sublist] for sublist in d_all['d_all']]

        # # f_all = [[[cv2.KeyPoint(x[0], x[1], x[2], x[3]) for x in inner_list] for inner_list in outer_list] for outer_list in f_all]
        # f_all = [[[cv2.KeyPoint(x[0], x[1], x[2], (np.degrees(x[3]) + 360) % 360) for x in inner_list] for inner_list in outer_list] for outer_list in f_all]
        # # # print("oe")

        # with open('f_all_big_dddd_105.pkl', 'rb') as f:
        #     f_all_serializable = pickle.load(f)
        # with open('d_all_big_dddd_105.pkl', 'rb') as f:
        #     d_all = pickle.load(f)

        # import pickle

        # with open(r"P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject105\Session496\montaged_corrected\shifts.pickle", 'rb') as f:
        #     shifts = pickle.load(f)

        # f_all = [[dict_to_keypoints(kp) for kp in sublist] for sublist in f_all_serializable]

        bestNumOkMatches = {}
        while np.sum(self.Matched) < self.N:
            refIndex = self.find_closest_unmatched_image()
            self.RelativeTransformToRef[:, :, refIndex] = np.eye(3)
            self.Matched[refIndex] = 1
            self.MatchedTo[refIndex] = refIndex
            bestRefIndex, bestTransform, bestNumOkMatches, bestNumMatches, searchLevel, \
                percentageThresh, stuckFlag = self.reset_variables()
            self.find_all_matches(f_all, d_all, refIndex, bestRefIndex, bestTransform,
                                  bestNumOkMatches, bestNumMatches, searchLevel,
                                  percentageThresh, stuckFlag)

        AllRefIndex, RefChains = self.find_reference_frames()
        TotalTransform, RefChains = self.calculate_total_transform(AllRefIndex, RefChains)
        CoMX, CoMY = self.calculate_centers_of_mass(RefChains, len(AllRefIndex))
        maxXRef, minXRef, maxYRef, minYRef, Xwidth, Ywidth = initialize_bounding_box(len(AllRefIndex))

        NumOfRefs = len(AllRefIndex)
        maxXRef, minXRef, maxYRef, minYRef, Xwidth, Ywidth = self.calculate_local_bounding_box(NumOfRefs,
                                                                                               RefChains,
                                                                                               refIndex,
                                                                                               TotalTransform,
                                                                                               maxXRef,
                                                                                               minXRef,
                                                                                               maxYRef,
                                                                                               minYRef)
        CoMX, CoMY = self.adjust_duplicate_com(NumOfRefs, CoMX, CoMY)
        refOrderX_I, refOrderY_I = self.find_relative_translation(CoMX, CoMY)
        refGlobalTransX, refGlobalTransY = self.find_total_translation(NumOfRefs)
        pad = 30
        maxWidthX = Xwidth[refOrderX_I[0]]
        maxWidthY = Ywidth[refOrderY_I[0]]

        maxXRef, minXRef, maxYRef, minYRef = self.adjust_bounding_box(NumOfRefs,
                                                                      CoMX,
                                                                      CoMY,
                                                                      maxXRef,
                                                                      minXRef,
                                                                      maxYRef,
                                                                      minYRef)
        refGlobalTransX, refGlobalTransY = self.calculate_global_translation(NumOfRefs,
                                                                             CoMX,
                                                                             CoMY,
                                                                             refOrderX_I,
                                                                             refOrderY_I,
                                                                             maxXRef,
                                                                             minXRef,
                                                                             maxYRef,
                                                                             minYRef,
                                                                             refGlobalTransX,
                                                                             refGlobalTransY,
                                                                             pad)
        TotalTransform = self.adjust_transformation(NumOfRefs,
                                                    refGlobalTransX,
                                                    refGlobalTransY,
                                                    RefChains,
                                                    TotalTransform)

        ur, vr, maxXAll, minXAll, maxYAll, minYAll = self.calculate_global_bounding_box(refIndex,
                                                                                        TotalTransform)
        outNameList = self.create_output_name_list(NumOfRefs, self.MN)
        fovlist = self.calculate_fov_list(self.pixelScale)
        tforms = self.create_tform_matrices(TotalTransform, minXAll, minYAll)
        tforms = self.adjust_tform_translations(tforms)

        # Loop over modalities
        for modality in range(self.MN):
            # Initialize blank combined image of all pieces for the modality
            combined_all = np.zeros((len(vr), len(ur)), dtype=np.uint8)

            for ref_index in range(NumOfRefs):
                if self.imageFilename[modality][AllRefIndex[ref_index]] is not None:
                    # Initialize blank combined image for the modality/piece
                    combined = np.zeros((len(vr), len(ur)), dtype=np.uint8)

                    ref = RefChains[ref_index][0]

                    for n in RefChains[ref_index]:
                        if self.imageFilename[modality][n] is not None:
                            im = self.read_and_transform_image(self.imageFilename[modality][n],
                                                               self.pixelScale[refIndex], tforms[n])

                            # Save each individually transformed image
                            _, name = os.path.split(self.imageFilename[modality][n])

                            im, nonzero = pad_image(im, vr, ur)

                            # Add to combined image
                            try:
                                combined[nonzero] = im[nonzero]
                            except IndexError:
                                # cut image to same shape as combined
                                im = im[:len(vr), :len(ur)]
                                nonzero = nonzero[:len(vr), :len(ur)]
                                combined[nonzero] = im[nonzero]

                            # Save to file labelled with the reference of the first image of the chain
                            save_filename = f"{name}_aligned_to_ref{ref}_m{modality}.tif"
                            self.save_image(im, nonzero, save_filename, self.outputDir)

                            numWritten += 1

                    # Add to all combined image
                    nonzero = combined > 0
                    combined_all[nonzero] = combined[nonzero]
                    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
                    combined = np.concatenate([combined,
                                               np.round(nonzero * 255).astype(np.uint8)[:, :, None]],
                                              axis=2)

                    # Save combined image for each piece - removed for memory concerns.
                    if NumOfRefs > 1:  # Only necessary if more than one piece
                        save_filename = f"ref_{ref}_combined_m{modality}.tif"
                        outNameList[modality][ref_index] = save_filename

                        self.save_image(combined, nonzero, save_filename, self.outputDir)

            # Save the combined image of all the pieces
            save_filename = f"all_ref_combined_m{modality}.tif"
            outNameList[0][modality] = save_filename

            nonzero = combined_all > 0
            combined_all = cv2.cvtColor(combined_all, cv2.COLOR_GRAY2RGB)
            combined_all = np.concatenate([combined_all,
                                           np.round(nonzero * 255).astype(np.uint8)[:, :, None]],
                                          axis=2)

            self.save_image(combined_all, nonzero, save_filename, self.outputDir)

    def initialize_variables(self):
        """
        Initialize the variables used in the image processing.
        """
        # stores relative transform between matched pairs
        self.RelativeTransformToRef = np.zeros((3, 3, self.N))
        self.Matched = np.zeros(self.N)  # store if image is matched already
        self.MatchedTo = np.zeros(self.N)  # keeps track of which image is matched to which

        # stores all pairwise transforms
        self.ResultsNumOkMatches = -np.ones((self.N, self.N))
        self.ResultsNumMatches = -np.ones((self.N, self.N))
        self.ResultsScaleToRef = -np.ones((self.N, self.N))
        self.ResultsTransformToRef = np.zeros((3, 3, self.N, self.N))

    def find_closest_unmatched_image(self) -> int:
        """
        Find the closest unmatched image to the origin and set it as the new reference frame.

        :return: The index of the closest unmatched image.
        :rtype: int
        """
        unMatchedI = np.where(self.Matched == 0)[0]
        if len(unMatchedI) > 1:  # check if there are any unmatched images
            unMatchedLocXY = self.LocXY[:, unMatchedI]
            sortUnMatchedLocXY = np.argsort(np.abs(unMatchedLocXY), axis=0)

            # set reference frame
            refIndex = unMatchedI[sortUnMatchedLocXY[0, 0]]
        else:
            refIndex = unMatchedI[0]
        return refIndex

    def reset_variables(self) -> Tuple[int, np.ndarray, int, int, int, float, int]:
        """
        Reset the variables to their initial values.

        :return: A tuple containing the initial values of the variables.
        :rtype: tuple
        """
        return 0, np.zeros((3, 3)), 0, 1000000, 5, 0.1, 0

    def find_all_matches(self, f_all: list, d_all: list, refIndex: int,
                         bestRefIndex: int, bestTransform: np.ndarray,
                         bestNumOkMatches: int, bestNumMatches: int,
                         searchLevel: int, percentageThresh: float,
                         stuckFlag: int):
        """
        Find all SIFT matches for the images.

        :param f_all: The features for all images.
        :type f_all: list
        :param d_all: The descriptors for all images.
        :type d_all: list
        :param refIndex: The index of the reference frame.
        :type refIndex: int
        :param bestRefIndex: The index of the best reference frame.
        :type bestRefIndex: int
        :param bestTransform: The best transformation matrix.
        :type bestTransform: np.ndarray
        :param bestNumOkMatches: The best number of good matches.
        :type bestNumOkMatches: int
        :param bestNumMatches: The best number of matches.
        :type bestNumMatches: int
        :param searchLevel: The search level.
        :type searchLevel: int
        :param percentageThresh: The percentage threshold for good matches.
        :type percentageThresh: float
        :param stuckFlag: The flag indicating if the process is stuck.
        :type stuckFlag: int
        """
        while stuckFlag == 0:
            stuckFlag = 1
            for n in range(self.N):
                if self.Matched[n] == 0 and not np.isnan(self.LocXY[0, n]):
                    bestNumOkMatches, bestNumMatches = \
                        self.find_best_match(f_all, d_all, n, refIndex, bestRefIndex,
                                             bestTransform, bestNumOkMatches, bestNumMatches,
                                             searchLevel)
            if bestNumOkMatches > self.NumOkMatchesThresh or \
                (bestNumOkMatches / bestNumMatches > percentageThresh and bestNumOkMatches > 1):
                self.RelativeTransformToRef[:, :, n] = bestTransform
                self.Matched[n] = 1
                self.MatchedTo[n] = bestRefIndex
                stuckFlag = 0

    def find_best_match(self, f_all: list, d_all: list, n: int, refIndex: int,
                        bestRefIndex: int, bestTransform: np.ndarray,
                        bestNumOkMatches: int, bestNumMatches: int,
                        searchLevel: int) -> Tuple[list, int]:
        """
        Find the best match for a given image.

        :param f_all: The features for all images.
        :type f_all: list
        :param d_all: The descriptors for all images.
        :type d_all: list
        :param n: The index of the current image.
        :type n: int
        :param refIndex: The index of the reference frame.
        :type refIndex: int
        :param bestRefIndex: The index of the best reference frame.
        :type bestRefIndex: int
        :param bestTransform: The best transformation matrix.
        :type bestTransform: np.ndarray
        :param bestNumOkMatches: The best number of good matches.
        :type bestNumOkMatches: int
        :param bestNumMatches: The best number of matches.
        :type bestNumMatches: int
        :param searchLevel: The search level.
        :type searchLevel: int
        """
        bestNumOkMatches = 0
        bestNumMatches = 1000000
        for refIndex in range(self.N):
            CurrentDistToRef = np.sum(np.abs(self.LocXY[:, refIndex] - self.LocXY[:, n]))
            if CurrentDistToRef <= searchLevel and refIndex != n:
                if self.Matched[refIndex] == 1:
                    bestRefIndex, bestTransform, bestNumOkMatches, bestNumMatches = \
                        self.check_match(f_all, d_all, n, refIndex, bestRefIndex,
                                         bestTransform, bestNumOkMatches, bestNumMatches)
        return bestNumOkMatches, bestNumMatches

    def check_match(self, f_all: list, d_all: list, n: int, refIndex: int,
                    bestRefIndex: int, bestTransform: np.ndarray,
                    bestNumOkMatches: int, bestNumMatches: int) -> tuple:
        """
        Check the match between the current image and the reference frame.

        :param f_all: The features for all images.
        :type f_all: list
        :param d_all: The descriptors for all images.
        :type d_all: list
        :param n: The index of the current image.
        :type n: int
        :param refIndex: The index of the reference frame.
        :type refIndex: int
        :param bestRefIndex: The index of the best reference frame.
        :type bestRefIndex: int
        :param bestTransform: The best transformation matrix.
        :type bestTransform: np.ndarray
        :param bestNumOkMatches: The best number of good matches.
        :type bestNumOkMatches: int
        :param bestNumMatches: The best number of matches.
        :type bestNumMatches: int

        :return: The index of the best reference frame,
            the best transformation matrix, the best number of good matches,
            and the best number of matches.
        :rtype: tuple
        """
        if self.ResultsNumOkMatches[n, refIndex] == -1:
            refImg = [None] * self.MN
            currentImg = [None] * self.MN
            for m in range(self.MN):
                im = cv2.imread(self.imageFilename[m][refIndex])
                new_size = (int(im.shape[0]*self.pixelScale[refIndex]),
                            int(im.shape[1]*self.pixelScale[refIndex]))
                refImg[m] = resize(img_as_float(im[:, :, 0]), new_size, mode='reflect')
                im = cv2.imread(self.imageFilename[m][n])
                new_size = (int(im.shape[0]*self.pixelScale[refIndex]),
                            int(im.shape[1]*self.pixelScale[refIndex]))
                currentImg[m] = resize(img_as_float(im[:, :, 0]), new_size, mode='reflect')

            relativeTransform, numOkMatches, numMatches, bestScale, scores = \
                sift_mosaic_fast_MultiModal(refImg,
                                            currentImg,
                                            [f[refIndex] for f in f_all],
                                            [d[refIndex] for d in d_all],
                                            [f[n] for f in f_all], [d[n] for d in d_all],
                                            None)
            # all_scores[imageFilename[m][refIndex]+'_'+imageFilename[m][n]] = scores
            self.ResultsNumOkMatches[n, refIndex] = numOkMatches
            self.ResultsNumMatches[n, refIndex] = numMatches
            self.ResultsTransformToRef[:, :, n, refIndex] = relativeTransform
            self.ResultsScaleToRef[n, refIndex] = bestScale

        # set as best if better than current best
        if self.ResultsNumOkMatches[n, refIndex] > bestNumOkMatches or \
            (self.ResultsNumOkMatches[n, refIndex] == bestNumOkMatches and \
                self.ResultsNumOkMatches[n, refIndex] / self.ResultsNumMatches[n, refIndex] >\
                    bestNumOkMatches / bestNumMatches):
            bestRefIndex = refIndex
            bestTransform[:, :] = self.ResultsTransformToRef[:, :, n, refIndex]
            bestNumOkMatches = self.ResultsNumOkMatches[n, refIndex]
            bestNumMatches = self.ResultsNumMatches[n, refIndex]
        return bestRefIndex, bestTransform, bestNumOkMatches, bestNumMatches

    def use_best_match(self, bestNumOkMatches: int, bestNumMatches: int,
                       percentageThresh: float) -> bool:
        """
        Determine whether to use the best match based on the number of good matches and
        the percentage threshold.

        :param bestNumOkMatches: The best number of good matches.
        :type bestNumOkMatches: int
        :param bestNumMatches: The best number of matches.
        :type bestNumMatches: int
        :param percentageThresh: The percentage threshold.
        :type percentageThresh: float

        :return: Whether to use the best match.
        :rtype: bool
        """
        if bestNumOkMatches > self.NumOkMatchesThresh or \
            (bestNumOkMatches / bestNumMatches > percentageThresh and bestNumOkMatches > 1):
            return True
        return False

    def sortUsingLocXY(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Sorts the given data based on the absolute values of the coordinates in LocXY.

        :param LocXY: The locations of the images.
        :type LocXY: np.ndarray
        :param inData: The input data.
        :type inData: np.ndarray
        :param imageFilename: The filenames of the images.
        :type imageFilename: np.ndarray
        :param pixelScale: The pixel scales of the images.
        :type pixelScale: np.ndarray
        :param ID: The IDs of the images.
        :type ID: list
        :param MN: The number of modalities.
        :type MN: int

        :return: The sorted locations of the images, the sorted input data,
            the sorted filenames of the images, the sorted pixel scales of the images,
            and the sorted IDs of the images.
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
        """
        I = np.lexsort((np.abs(self.LocXY[1, :]), np.abs(self.LocXY[0, :])))
        self.LocXY = self.LocXY[:, I]
        self.pixelScale = np.array(self.pixelScale)[I]

        for m in range(self.MN):
            self.imageFilename[m] = [self.imageFilename[m][i] for i in I]
            self.inData[m] = [self.inData[m][i] for i in I]

    def find_reference_frames(self) -> Tuple[list, list]:
        """
        Find the reference frames in the set of matched images.

        :return: A list of indices of the reference frames and a list of empty lists
            with the same length as the number of reference frames.
        :rtype: (list, list)
        """
        AllRefIndex = [n for n in range(self.N) if self.MatchedTo[n] == n]
        NumOfRefs = len(AllRefIndex)
        RefChains = [[] for _ in range(NumOfRefs)]
        return AllRefIndex, RefChains

    def calculate_total_transform(self, AllRefIndex: list,
                                  RefChains: list) -> Tuple[np.ndarray, list]:
        """
        Calculates the total transformation matrix for each image.

        :param AllRefIndex: The indices of all reference frames.
        :type AllRefIndex: list
        :param RefChains: The reference chains for all images.
        :type RefChains: list

        :return: The total transformation matrices for all images and the
            updated reference chains.
        :rtype: tuple (np.ndarray, list)
        """
        TotalTransform = np.zeros_like(self.RelativeTransformToRef)
        for n in range(self.N):
            H = self.RelativeTransformToRef[:, :, n]
            nextRef = self.MatchedTo[n]
            while nextRef != self.MatchedTo[int(nextRef)]:
                nextH = self.RelativeTransformToRef[:, :, int(nextRef)]
                H = H @ nextH
                nextRef = self.MatchedTo[int(nextRef)]
            ind = AllRefIndex.index(nextRef)
            RefChains[ind].append(n)
            TotalTransform[:, :, n] = H
        return TotalTransform, RefChains

    def calculate_centers_of_mass(self, RefChains: list,
                                  NumOfRefs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the centers of mass for all reference chains.

        :param RefChains: The reference chains for all images.
        :type RefChains: list
        :param NumOfRefs: The number of reference frames.
        :type NumOfRefs: int

        :return: The x-coordinates and the y-coordinates of the centers of mass
            for all reference chains.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        CoMX = np.array([np.median(self.LocXY[0, RefChains[i]]) for i in range(NumOfRefs)])
        CoMY = np.array([np.median(self.LocXY[1, RefChains[i]]) for i in range(NumOfRefs)])
        return CoMX, CoMY

    def calculate_local_bounding_box(self, NumOfRefs: int, RefChains: list,
                                     refIndex: int, TotalTransform: np.ndarray,
                                     maxXRef: np.ndarray, minXRef: np.ndarray,
                                     maxYRef: np.ndarray, minYRef: np.ndarray) -> \
                                     Tuple[np.ndarray]:
        """
        Calculates the local bounding box for each reference chain.

        :param NumOfRefs: The number of reference frames.
        :type NumOfRefs: int
        :param RefChains: The reference chains for all images.
        :type RefChains: list
        :param refIndex: The index of the reference frame.
        :type refIndex: int
        :param TotalTransform: The total transformation matrices for all images.
        :type TotalTransform: np.ndarray
        :param maxXRef: The maximum x-coordinates of the bounding boxes for all reference frames.
        :type maxXRef: np.ndarray
        :param minXRef: The minimum x-coordinates of the bounding boxes for all reference frames.
        :type minXRef: np.ndarray
        :param maxYRef: The maximum y-coordinates of the bounding boxes for all reference frames.
        :type maxYRef: np.ndarray
        :param minYRef: The minimum y-coordinates of the bounding boxes for all reference frames.
        :type minYRef: np.ndarray

        :return: The maximum and minimum x-coordinates and y-coordinates of the
            bounding boxes for all reference frames and the widths of the bounding
            boxes in the x-direction and y-direction for all reference frames.
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
        Xwidth = np.zeros(NumOfRefs)
        Ywidth = np.zeros(NumOfRefs)
        for i in range(NumOfRefs):
            for n in RefChains[i]:
                if self.imageFilename[0][n] is not None:
                    im = cv2.imread(self.imageFilename[0][n])
                    new_size = (int(im.shape[0]*self.pixelScale[refIndex]),
                                int(im.shape[1]*self.pixelScale[refIndex]))
                    im = resize(img_as_float(im[:, :, 0]), new_size, mode='reflect')
                    H = TotalTransform[:, :, n]
                    box = np.array([[1, im.shape[1], im.shape[1], 1],
                                    [1, 1, im.shape[0], im.shape[0]],
                                    [1, 1, 1, 1]])
                    box_ = pinv(H) @ box
                    box_[0, :] /= box_[2, :]
                    box_[1, :] /= box_[2, :]
                    maxXRef[i] = max(maxXRef[i], np.max(box_[0, :]))
                    minXRef[i] = min(minXRef[i], np.min(box_[0, :]))
                    maxYRef[i] = max(maxYRef[i], np.max(box_[1, :]))
                    minYRef[i] = min(minYRef[i], np.min(box_[1, :]))
            Xwidth[i] = maxXRef[i] - minXRef[i]
            Ywidth[i] = maxYRef[i] - minYRef[i]
        return maxXRef, minXRef, maxYRef, minYRef, Xwidth, Ywidth

    def calculate_global_bounding_box(self, refIndex: int,
                                      TotalTransform: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates the global bounding box for all images.

        :param refIndex: The reference index for pixel scaling.
        :type refIndex: int
        :param TotalTransform: The total transformation for all images.
        :type TotalTransform: np.ndarray

        :return: The ranges of u and v, and the maximum and minimum x and y
            coordinates of the global bounding box.
        :rtype: tuple of np.ndarray and float
        """
        maxXAll, minXAll, maxYAll, minYAll = self.initialize_global_variables()
        for n in range(self.N):
            if self.imageFilename[0][n] is not None:
                im = cv2.imread(self.imageFilename[0][n])
                new_size = (int(im.shape[0]*self.pixelScale[refIndex]),
                            int(im.shape[1]*self.pixelScale[refIndex]))
                im = resize(img_as_float(im[:, :, 0]), new_size, mode='reflect')
                H = TotalTransform[:, :, n]

                # Transform the 4 corners
                box = np.array([[1, im.shape[0], im.shape[0], 1],
                                [1, 1, im.shape[1], im.shape[1]],
                                [1, 1, 1, 1]])
                box_ = np.linalg.pinv(H).dot(box)
                box_[0, :] /= box_[2, :]
                box_[1, :] /= box_[2, :]

                maxXAll = max(maxXAll, np.max(box_[0, :]))
                minXAll = min(minXAll, np.min(box_[0, :]))
                maxYAll = max(maxYAll, np.max(box_[1, :]))
                minYAll = min(minYAll, np.min(box_[1, :]))

        ur = np.arange(minXAll, maxXAll+1)
        vr = np.arange(minYAll, maxYAll+1)
        return ur, vr, maxXAll, minXAll, maxYAll, minYAll

def read_and_transform_image(filename: str, pixel_scale: float,
                                tforms: np.ndarray) -> np.ndarray:
    """
    Read an image from a file, resize it according to the pixel scale, and apply
    a transformation.

    :param filename: The name of the file from which to read the image.
    :type filename: str
    :param pixel_scale: The scale factor to apply to the size of the image.
    :type pixel_scale: float
    :param tforms: The transformation matrix to apply to the image.
    :type tforms: np.ndarray
    :return: The transformed image.
    :rtype: numpy.ndarray
    """
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    new_size = (int(im.shape[0]*pixel_scale), int(im.shape[1]*pixel_scale))
    im = resize(im, new_size, mode='reflect')

    p1 = np.array([0, 0, 1])
    p2 = np.array([0, im.shape[:2][1], 1])
    dy = tforms[0, 2]
    tforms[0, 2] = tforms[1, 2]
    tforms[1, 2] = dy
    p1_transformed = np.array(tforms) @ p1
    p2_transformed = np.array(tforms) @ p2
    p1_transformed = p1_transformed[:2] / p1_transformed[2]
    p2_transformed = p2_transformed[:2] / p2_transformed[2]
    im, _ = shift_image(p1_transformed, p2_transformed, im)

    return im

def save_image(im, nonzero, filename, output_dir):
    """
    Save an image to a file.

    :param im: The image to save.
    :type im: numpy.ndarray
    :param nonzero: A mask indicating the non-zero pixels in the image.
    :type nonzero: numpy.ndarray
    :param filename: The name of the file to which to save the image.
    :type filename: str
    :param output_dir: The directory in which to save the file.
    :type output_dir: str
    """
    if im.dtype == np.float64 or im.dtype == np.float32:
        im = np.round(im * 255).astype(np.uint8)
    elif im.dtype != np.uint8:
        im = im.astype(np.uint8)

    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = np.concatenate([im, np.round(nonzero * 255).astype(np.uint8)[:, :, None]], axis=2)
    cv2.imwrite(os.path.join(output_dir, filename), im)

def adjust_duplicate_com(NumOfRefs: int, CoMX: np.ndarray,
                         CoMY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusts the centers of mass for all reference chains to avoid duplicates.

    :param NumOfRefs: The number of reference frames.
    :type NumOfRefs: int
    :param CoMX: The x-coordinates of the centers of mass for all reference chains.
    :type CoMX: np.ndarray
    :param CoMY: The y-coordinates of the centers of mass for all reference chains.
    :type CoMY: np.ndarray

    :return: The adjusted x-coordinates and y-coordinates of the centers of mass
        for all reference chains.
    :rtype: tuple (np.ndarray, np.ndarray)
    """
    ee = .0001
    checked = np.zeros(NumOfRefs, dtype=bool)
    for s in range(NumOfRefs):
        if not checked[s]:
            counter = 0
            checked[s] = True
            for ss in range(NumOfRefs):
                if CoMX[s] == CoMX[ss] and CoMY[s] == CoMY[ss]:
                    checked[ss] = True
                    if CoMX[ss] >= CoMY[ss]:
                        CoMX[ss] += ee * counter
                    else:
                        CoMY[ss] += ee * counter
                    counter += 1
    return CoMX, CoMY

def find_relative_translation(CoMX: np.ndarray, CoMY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the relative translation between the centers of mass for all reference chains.

    :param CoMX: The x-coordinates of the centers of mass for all reference chains.
    :type CoMX: np.ndarray
    :param CoMY: The y-coordinates of the centers of mass for all reference chains.
    :type CoMY: np.ndarray

    :return: The indices that would sort the x-coordinates and y-coordinates of
        the centers of mass for all reference chains.
    :rtype: tuple (np.ndarray, np.ndarray)
    """
    refOrderX_I = np.argsort(CoMX)
    # refOrderX = np.sort(CoMX)
    refOrderY_I = np.argsort(CoMY)[::-1]
    # refOrderY = np.sort(CoMY)[::-1]
    return refOrderX_I, refOrderY_I

def find_total_translation(NumOfRefs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes the total translation in the x-direction and y-direction for all reference frames.

    :param NumOfRefs: The number of reference frames.
    :type NumOfRefs: int

    :return: The total translation in the x-direction and y-direction for all reference frames.
    :rtype: tuple (np.ndarray, np.ndarray)
    """
    refGlobalTransX = np.zeros(NumOfRefs)
    refGlobalTransY = np.zeros(NumOfRefs)
    return refGlobalTransX, refGlobalTransY

def adjust_bounding_box(NumOfRefs: int, CoMX: np.ndarray, CoMY: np.ndarray,
                        maxXRef: np.ndarray, minXRef: np.ndarray, maxYRef: np.ndarray,
                        minYRef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adjusts the bounding box for all reference frames based on their centers of mass.

    :param NumOfRefs: The number of reference frames.
    :type NumOfRefs: int
    :param CoMX: The x-coordinates of the centers of mass for all reference frames.
    :type CoMX: np.ndarray
    :param CoMY: The y-coordinates of the centers of mass for all reference frames.
    :type CoMY: np.ndarray
    :param maxXRef: The maximum x-coordinates of the bounding box for all reference frames.
    :type maxXRef: np.ndarray
    :param minXRef: The minimum x-coordinates of the bounding box for all reference frames.
    :type minXRef: np.ndarray
    :param maxYRef: The maximum y-coordinates of the bounding box for all reference frames.
    :type maxYRef: np.ndarray
    :param minYRef: The minimum y-coordinates of the bounding box for all reference frames.
    :type minYRef: np.ndarray

    :return: The adjusted maximum and minimum x-coordinates and y-coordinates of the
        bounding box for all reference frames.
    :rtype: tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    for i in range(NumOfRefs):
        ind = np.where(CoMX == CoMX[i])[0]
        maxXRef[ind] = np.max(maxXRef[ind])
        minXRef[ind] = np.min(minXRef[ind])

        ind = np.where(CoMY == CoMY[i])[0]
        maxYRef[ind] = np.max(maxYRef[ind])
        minYRef[ind] = np.min(minYRef[ind])

    minXRef[np.isnan(CoMX)] = np.min(minXRef[~np.isnan(CoMX)])
    minYRef[np.isnan(CoMY)] = np.min(minYRef[~np.isnan(CoMY)])

    maxXRef[np.isnan(maxXRef)] = np.max(maxXRef[~np.isnan(CoMX)])
    maxYRef[np.isnan(CoMY)] = np.max(maxYRef[~np.isnan(CoMY)])
    return maxXRef, minXRef, maxYRef, minYRef

def initialize_global_variables() -> Tuple[int, int, int, int]:
    """
    Initializes the global variables for the maximum and minimum x-coordinates
    and y-coordinates.

    :return: The initialized maximum and minimum x-coordinates and y-coordinates.
    :rtype: tuple (int, int, int, int)
    """
    maxXAll = -1000000000
    minXAll = 1000000000
    maxYAll = -1000000000
    minYAll = 1000000000
    return maxXAll, minXAll, maxYAll, minYAll

def calculate_global_translation(NumOfRefs: int, CoMX: np.ndarray, CoMY: np.ndarray,
                                 refOrderX_I: np.ndarray, refOrderY_I: np.ndarray,
                                 maxXRef: np.ndarray, minXRef: np.ndarray,
                                 maxYRef: int, minYRef: int,
                                 refGlobalTransX: np.ndarray, refGlobalTransY: np.ndarray,
                                 pad: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the global translation in the x-direction and y-direction for all
    reference frames based on their centers of mass.

    :param NumOfRefs: The number of reference frames.
    :type NumOfRefs: int
    :param CoMX: The x-coordinates of the centers of mass for all reference frames.
    :type CoMX: np.ndarray
    :param CoMY: The y-coordinates of the centers of mass for all reference frames.
    :type CoMY: np.ndarray
    :param refOrderX_I: The indices that would sort the x-coordinates of the centers
        of mass for all reference frames.
    :type refOrderX_I: np.ndarray
    :param refOrderY_I: The indices that would sort the y-coordinates of the centers
        of mass for all reference frames.
    :type refOrderY_I: np.ndarray
    :param maxXRef: The maximum x-coordinates of the bounding box for all reference frames.
    :type maxXRef: int
    :param minXRef: The minimum x-coordinates of the bounding box for all reference frames.
    :type minXRef: int
    :param maxYRef: The maximum y-coordinates of the bounding box for all reference frames.
    :type maxYRef: int
    :param minYRef: The minimum y-coordinates of the bounding box for all reference frames.
    :type minYRef: int
    :param refGlobalTransX: The total translation in the x-direction for all reference frames.
    :type refGlobalTransX: np.ndarray
    :param refGlobalTransY: The total translation in the y-direction for all reference frames.
    :type refGlobalTransY: np.ndarray
    :param pad: The padding to be added to the total translation in the x-direction and
        y-direction for all reference frames.
    :type pad: int

    :return: The calculated global translation in the x-direction and y-direction for all
        reference frames.
    :rtype: tuple (np.ndarray, np.ndarray)
    """
    for s in range(1, NumOfRefs):
        if not np.isnan(CoMX[refOrderX_I[s]]):
            if CoMX[refOrderX_I[s]] == CoMX[refOrderX_I[s-1]]:
                refGlobalTransX[refOrderX_I[s]] = refGlobalTransX[refOrderX_I[s-1]]
            else:
                refGlobalTransX[refOrderX_I[s]] = refGlobalTransX[refOrderX_I[s-1]] +\
                    maxXRef[refOrderX_I[s-1]] - minXRef[refOrderX_I[s]] + pad

        if not np.isnan(CoMY[refOrderY_I[s]]):
            if CoMY[refOrderY_I[s]] == CoMY[refOrderY_I[s-1]]:
                refGlobalTransY[refOrderY_I[s]] = refGlobalTransY[refOrderY_I[s-1]]
            else:
                refGlobalTransY[refOrderY_I[s]] = refGlobalTransY[refOrderY_I[s-1]] +\
                    maxYRef[refOrderY_I[s-1]] - minYRef[refOrderY_I[s]] + pad
    return refGlobalTransX, refGlobalTransY

def adjust_transformation(NumOfRefs: int, refGlobalTransX: np.ndarray,
                          refGlobalTransY: np.ndarray, RefChains: list,
                          TotalTransform: np.ndarray) -> np.ndarray:
    """
    Adjusts the total transformation for all reference frames based on the global
    translation in the x-direction and y-direction.

    :param NumOfRefs: The number of reference frames.
    :type NumOfRefs: int
    :param refGlobalTransX: The total translation in the x-direction for all reference frames.
    :type refGlobalTransX: np.ndarray
    :param refGlobalTransY: The total translation in the y-direction for all reference frames.
    :type refGlobalTransY: np.ndarray
    :param RefChains: The chains of reference frames.
    :type RefChains: list of list of int
    :param TotalTransform: The total transformation for all reference frames.
    :type TotalTransform: np.ndarray

    :return: The adjusted total transformation for all reference frames.
    :rtype: np.ndarray
    """
    for i in range(NumOfRefs):
        refGlobalTrans = np.eye(3)
        refGlobalTrans[0, 2] = -refGlobalTransX[i]
        refGlobalTrans[1, 2] = -refGlobalTransY[i]
        for n in RefChains[i]:
            TotalTransform[:, :, n] = np.dot(TotalTransform[:, :, n], refGlobalTrans)
    return TotalTransform

def create_output_name_list(NumOfRefs: int, MN: int) -> list:
    """
    Creates a list of output names based on the number of reference frames and modalities.

    :param NumOfRefs: The number of reference frames.
    :type NumOfRefs: int
    :param MN: The number of modalities.
    :type MN: int

    :return: The list of output names.
    :rtype: list of None or list of list of None
    """
    if NumOfRefs == 1:
        outNameList = [None] * MN  # Just 1 output for each modality if only one piece
    else:
        # Otherwise one for each piece and extra one for all the pieces combined
        outNameList = [[None] * (NumOfRefs+1) for _ in range(MN)]
    return outNameList

def calculate_fov_list(pixelScale: np.ndarray) -> List[str]:
    """
    Calculates the list of unique field of view (FOV) values from the given pixel scale.

    :param pixelScale: The pixel scale of the images.
    :type pixelScale: np.ndarray

    :return: The list of unique FOV values, rounded to two decimal places and
        converted to strings.
    :rtype: list of str
    """
    fovlist = np.unique(pixelScale)
    fovlist = [f"{round(n, 2):.2f}" for n in fovlist]
    return fovlist

def create_tform_matrices(TotalTransform: np.ndarray, minXAll: float,
                          minYAll: float) -> np.ndarray:
    """
    Creates transformation matrices by adapting the total transformation for
    each image based on the minimum x and y coordinates.

    :param TotalTransform: The total transformation for all images.
    :type TotalTransform: np.ndarray
    :param minXAll: The minimum x coordinate of the global bounding box.
    :type minXAll: float
    :param minYAll: The minimum y coordinate of the global bounding box.
    :type minYAll: float

    :return: The transformation matrices for all images.
    :rtype: np.ndarray
    """
    # Determine the dominant direction of each shifted image
    Global = np.array([[1, 0, 0], [0, 1, 0], [-minXAll, -minYAll, 1]])
    # Adapt the transforms
    tforms = []

    # Create the tform matrices
    for n in range(TotalTransform.shape[2]):
        H = TotalTransform[:, :, n]
        tform = H @ Global.T
        tforms.append(tform)

    # Convert the list of tform matrices to a 3D numpy array
    tforms = np.array(tforms)
    return tforms

def adjust_tform_translations(tforms: np.ndarray) -> np.ndarray:
    """
    Adjusts the x and y translations in the transformation matrices to ensure
    they are non-negative.

    :param tforms: The transformation matrices for all images.
    :type tforms: np.ndarray

    :return: The adjusted transformation matrices.
    :rtype: np.ndarray
    """
    # Find the minimum x and y translations
    tforms[:, 0, 2] = -tforms[:, 0, 2]
    tforms[:, 1, 2] = -tforms[:, 1, 2]
    min_x_translation = min(tforms[:, 0, 2])
    min_y_translation = min(tforms[:, 1, 2])

    if min_x_translation < 0:
        # Adjust the x and y translations
        for k in range(tforms.shape[0]):
            tforms[k, 0, 2] -= min_x_translation
    if min_y_translation < 0:
        for k in range(tforms.shape[0]):
            tforms[k, 1, 2] -= min_y_translation
    return tforms
