from typing import Tuple
from dataclasses import dataclass, field
import numpy as np
import scipy.interpolate

from src.shared.helpers.direction import Direction

@dataclass
class Layer:
    """
    A class to represent a layer with various thickness values and calculations.

    :param name: The name of the layer.
    :type name: str
    :param thickness_values: The list of thickness values.
    :type thickness_values: list
    :param padded_thickness_values: The list of padded thickness values.
    :type padded_thickness_values: list
    :param thickness_per_distance_X: The dictionary of thickness per distance in the x direction.
    :type thickness_per_distance_X: dict
    :param mean_thickness_per_distance_X: The dictionary of mean thickness per distance in the x direction.
    :type mean_thickness_per_distance_X: dict
    :param smoothed_thickness_per_distance_X: The dictionary of smoothed thickness per distance in the x direction.
    :type smoothed_thickness_per_distance_X: dict
    :param thickness_per_distance_Y: The dictionary of thickness per distance in the y direction.
    :type thickness_per_distance_Y: dict
    :param mean_thickness_per_distance_Y: The dictionary of mean thickness per distance in the y direction.
    :type mean_thickness_per_distance_Y: dict
    :param smoothed_thickness_per_distance_Y: The dictionary of smoothed thickness per distance in the y direction.
    :type smoothed_thickness_per_distance_Y: dict
    """
    name: str
    centerx: int = 0
    thickness_values: list = field(default_factory=list)
    padded_thickness_values: list = field(default_factory=list)
    thickness_per_distance_X: dict = field(default_factory=dict)
    mean_thickness_per_distance_X: dict = field(default_factory=dict)
    smoothed_thickness_per_distance_X: dict = field(default_factory=dict)
    thickness_per_distance_Y: dict = field(default_factory=dict)
    mean_thickness_per_distance_Y: dict = field(default_factory=dict)
    smoothed_thickness_per_distance_Y: dict = field(default_factory=dict)
    central_X : dict = field(default_factory=dict)
    central_Y : dict = field(default_factory=dict)

    def __getattr__(self, name, value):
        """
        Get the attribute value if the attribute is not found.

        :param name: The name of the attribute to get.
        :type name: str
        :return: The attribute value.
        :rtype: Any

        """
        if name == 'name':
            return self.name
        elif name == 'centerx':
            return self.centerx
        elif name == 'thickness_values':
            return self.thickness_values
        elif name == 'padded_thickness_values':
            return self.padded_thickness
        
        elif name == 'thickness_per_distance_X':
            return self.thickness_per_distance_X
        elif name == 'mean_thickness_per_distance_X':
            return self.mean_thickness_per_distance_X
        elif name == 'smoothed_thickness_per_distance_X':
            return self.smoothed_thickness_per_distance_X
        elif name == 'thickness_per_distance_Y':
            return self.thickness_per_distance_Y
        elif name == 'mean_thickness_per_distance_Y':
            return self.mean_thickness_per_distance_Y
        elif name == 'smoothed_thickness_per_distance_Y':
            return self.smoothed_thickness_per_distance_Y
        elif name == 'central_X':
            return self.central_X
        elif name == 'central_Y':
            return self.central_Y

        else:
            print(f'Layer object has no attribute {name}, returning set Value')
            return None

        


    def is_empty(self) -> bool:
        """
        Check if the layer is empty (defined by the absence of thickness values).

        :return: True if the layer is empty, False otherwise.
        :rtype: bool
        """
        return len(self.thickness_values) == 0 and len(self.padded_thickness_values) == 0

    def append_thickness(self, value):
        """
        Append a thickness value to the thickness_values list.

        :param value: The thickness value to append.
        :type value: Any
        """
        # print("DEBUG: appending thickness value, called by Layer in file layer.py")
        # print(f"DEBUG: value: {value}")

        # print (f"DEBUG: max thickness_values: {max(self.thickness_values)}")
        # print (f"DEBUG: min thickness_values: {min(self.thickness_values)}")

        # print("DEBUG:plotting thickness values before appending")
        # import matplotlib.pyplot as plt
        # plt.plot(self.thickness_values)
        # plt.show()

        self.thickness_values.append(value)

        # print (f"DEBUG: max thickness_values after adding : {max(self.thickness_values)}")
        # print (f"DEBUG: min thickness_values after adding : {min(self.thickness_values)}")


        # print("DEBUG:plotting thickness values after appending")
        # plt.plot(self.thickness_values)
        # plt.show()

    def calculate_mean_thickness_per_distance(self) -> Tuple[dict, dict]:
        """
        Calculate the mean thickness per distance for both x and y directions.

        This method calculates the mean thickness per distance for the x and y directions
        and updates the corresponding dictionaries.

        :return: A tuple containing the mean thickness per distance dictionaries for x and y directions.
        :rtype: Tuple[dict, dict]
        """
        mean = lambda x: np.mean(x) if len(x) > 0 else np.nan
        self.mean_thickness_per_distance_X = {
           ecc: mean(thicknesses) for ecc, thicknesses in self.thickness_per_distance_X.items()
        }

        self.mean_thickness_per_distance_Y = {
            ecc: mean(thicknesses) for ecc, thicknesses in self.thickness_per_distance_Y.items()
        }

        return self.mean_thickness_per_distance_X, self.mean_thickness_per_distance_Y

    def fill_gaps(self, step: float, round_step: int) -> None:
        """
        Fill small gaps (of size up to `4*step`) in the layer thickness values by linear interpolation.

        :param step: The step size for the interpolation.
        :type step: float
        :param round_step: The rounded step size for the interpolation.
        :type round_step: int
        """
        gap_size = 4 * step
        directions = [direction for direction in Direction if len(getattr(self, f'mean_thickness_per_distance_{direction.value}')) > 0]
        for direction in directions:
            dict_mtpd = getattr(self, f'mean_thickness_per_distance_{direction.value}')
            mtpd = np.array(list(dict_mtpd.items()))
            mtpd = mtpd[np.argsort(mtpd[:, 0])]
            diff = np.round(np.diff(mtpd[:,0]), round_step)
            if np.where((step < diff) & (diff <= gap_size))[0].size == 0:
                continue # in this case, there are no small gaps to be filled
            interp_func = scipy.interpolate.interp1d(mtpd[:, 0], mtpd[:, 1], kind='linear', fill_value="extrapolate")
            large_gaps_indices = np.where(diff > gap_size)[0] # find gaps larger than 4*step; those are not filled
            ecc_min = min(mtpd[:, 0])
            ecc_max = max(mtpd[:, 0])
            eccs = np.round(np.arange(ecc_min, ecc_max+step, step), round_step)
            start_large_gaps = mtpd[large_gaps_indices,0]
            end_large_gaps = mtpd[large_gaps_indices + 1,0]
            valid_eccs_mask = np.all([(eccs <= start) | (end <= eccs) for start, end in zip(start_large_gaps, end_large_gaps)], axis=0) # eccs outside of large gaps
            valid_eccs_mask &= (ecc_min <= eccs) & (eccs <= ecc_max)
            setattr(self, 
                    f'mean_thickness_per_distance_{direction.value}',
                    dict(zip(
                        eccs[valid_eccs_mask],
                        interp_func(eccs[valid_eccs_mask])
                    ))
            )
