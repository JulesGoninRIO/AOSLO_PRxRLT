from typing import List, Dict, Tuple
import numpy as np


def find_chain_centers(matched_chains: Dict[int, List[int]], \
                       locations: List[int]) -> Dict[int, np.ndarray]:
    """
    Find the degrees of each image

    :param matched_chains: the chains from the montaged image
    :type matched_chains: Dict[int, List[int]]
    :param locations: the list of locations present
    :type locations: List[int]
    :return: the centers of the images location in the montage
    :rtype: Dict[int, np.ndarray]
    """
    chain_centers: Dict[int, np.ndarray] = {}
    for parent in matched_chains:
        chain = matched_chains[parent]

        top_degree = 100.
        bottom_degree = -100.
        left_degree = 100.
        right_degree = -100.

        for ind in chain:
            top_degree = min(top_degree, locations[ind][1])
            bottom_degree = max(bottom_degree, locations[ind][1])
            left_degree = min(left_degree, locations[ind][0])
            right_degree = max(right_degree, locations[ind][0])
        chain_centers[parent] = np.array([top_degree + bottom_degree, left_degree + right_degree]) / 2
    return chain_centers

def find_matched_chains(matched_to: List[str]) -> Dict[int, List[int]]:
    """
    Find the matches from a List of conencted components
    """
    matched_chains: Dict[int, List[int]] = {}
    included_in_chains = np.zeros(matched_to.size, dtype=bool)
    for ind, parent in enumerate(matched_to):
        if parent == ind:
            matched_chains[ind] = [ind]
            included_in_chains[ind] = True

    while not np.all(included_in_chains):
        for ind, parent in enumerate(matched_to):
            if not included_in_chains[ind]:
                for chain_parent in matched_chains:
                    if parent in matched_chains[chain_parent]:
                        matched_chains[chain_parent].append(ind)
                        included_in_chains[ind] = True
                        break
    return matched_chains

def find_matched_chains(matched_to: List[str]) -> Dict[int, List[int]]:
    """
    Find the matches from an array of a List of connected components

    :param matched_to: the matched to connected components
    :type matched_to: List[str]
    :return: A dictionnary with the indice of the first element in the chain, and
            all the indices of the elements that are from the same component
    :rtype: Dict[int, List[int]]
    """

    matched_chains: Dict[int, List[int]] = {}
    included_in_chains = np.zeros(matched_to.size, dtype=bool)
    for ind, parent in enumerate(matched_to):
        if parent == ind:
            matched_chains[ind] = [ind]
            included_in_chains[ind] = True

    while not np.all(included_in_chains):
        for ind, parent in enumerate(matched_to):
            if not included_in_chains[ind]:
                for chain_parent in matched_chains:
                    if parent in matched_chains[chain_parent]:
                        matched_chains[chain_parent].append(ind)
                        included_in_chains[ind] = True
                        break
    return matched_chains