import logging
import os
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from scipy import io
from pathlib import Path


class MatlabReader():
    """
    Class that handles the Matlab file outputed by the montage's code
    """

    def __init__(self, montaged_path: Path) -> None:
        """
        Initialize the data from the file

        :param montaged_path: the directory where the file is
        :type montaged_path: Path
        """
        logging.info('Reading montaging data...')
        self.__montaged_path = montaged_path
        self.__filename = self.__montaged_path / 'AOMontageSave.mat'

        try:
            self.data = io.loadmat(self.__filename)
        except FileNotFoundError:
            logging.error(f"Unable to read the Matlab file: {self.__filename}. \
                Please check that the file exists. \
                If the error persists, re-run the Matlab's montage step.")
            raise
        except TypeError:
            # sometimes there is a *** TypeError: Expecting matrix here
            logging.error(f"Unable to read the Matlab file: {self.__filename}. \
                Please check that the file is not corrupted. \
                If the error persists, re-run the Matlab's montage step.")
            raise

        # Get the informations on which image belongs to which component

        # matched_to is the array with the values corresponding to the indices of
        # its matched element in the list
        self.matched_to = self.data['MatchedTo'].flatten() - 1
        # since in Matlab numeration starts from 1
        self.names = self.data['inData'][0]
        self.matched_chains = self.find_matched_chains()
        self.locations = np.transpose(self.data['LocXY'])

    def get_transforms(self) -> np.ndarray:
        """
        Get the total transforms from the data.

        This method retrieves the 'TotalTransform' array from the data, transposes it, and converts it to a float32 numpy array.

        :return: The transposed and type-casted 'TotalTransform' array.
        :rtype: np.ndarray
        """
        return self.data['TotalTransform'].transpose((2, 0, 1)).astype(np.float32)

    def find_matched_chains(self) -> Dict[int, List[int]]:
        """
        Find the matches from an array of a List of connected components
        matched_chains has the form [0: [0, 1, 2], 3: [3,4]]
        where it means that image at index 0, 1, 2 have been found to be overlapping
        with the image at index 0 thus forming a component

        :raises ValueError: if the match value is impossible
        :return: A dictionnary with the indice of the first element in the chain, and
                all the indices of the elements that are from the same component
        :rtype: Dict[int, List[int]]
        """

        if np.any([i >= self.matched_to.shape[0] for i in self.matched_to]):
            raise ValueError(
                f"{self.matched_to} cannot contain values greater than the"
                "size of the array because it does not match any other"
                "value"
            )

        matched_chains: Dict[int, List[int]] = {}
        included_in_chains = np.zeros(self.matched_to.size, dtype=bool)
        for ind, parent in enumerate(self.matched_to):
            if parent == ind:
                matched_chains[ind] = [ind]
                included_in_chains[ind] = True

        while not np.all(included_in_chains):
            for ind, parent in enumerate(self.matched_to):
                if not included_in_chains[ind]:
                    for chain_parent in matched_chains:
                        if parent in matched_chains[chain_parent]:
                            matched_chains[chain_parent].append(ind)
                            included_in_chains[ind] = True
                            break

        return matched_chains
