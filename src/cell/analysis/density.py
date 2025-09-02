from dataclasses import dataclass, field
from typing import Dict
from deprecated import deprecated

@dataclass
class Density:
    """
    A class to represent density data in X and Y directions.

    This class stores density data in the X and Y directions, as well as smoothed and fitted versions of the data.
    
    General reminder:
        
        - Negative X should correspond to the Temporal side
        
        - Positive X should correspond to the Nasal side
        
        - Negative Y should correspond to the Superior side
        
        - Positive Y should correspond to the Inferior side
        
    :param X: The dictionary of density data in the X direction.
    :type X: Dict[float,float]
    :param Y: The dictionary of density data in the Y direction.
    :type Y: Dict[float,float]
    :param X_smoothed: The dictionary of smoothed density data in the X direction.
    :type X_smoothed: Dict[float,float]
    :param Y_smoothed: The dictionary of smoothed density data in the Y direction.
    :type Y_smoothed: Dict[float,float]
    :param X_fitted: The dictionary of fitted density data in the X direction.
    :type X_fitted: Dict[float,float]
    :param Y_fitted: The dictionary of fitted density data in the Y direction.
    :type Y_fitted: Dict[float,float] 
    """
    X: Dict[float,float] = field(default_factory=dict)
    Y: Dict[float,float] = field(default_factory=dict)
    X_smoothed: Dict[float,float] = field(default_factory=dict)
    Y_smoothed: Dict[float,float] = field(default_factory=dict)
    X_fitted: Dict[float,float] = field(default_factory=dict)
    Y_fitted: Dict[float,float] = field(default_factory=dict)

    @deprecated
    def add_density(self, direction: str, distance: float, density: float):
        """
        [LEGACY] Add density data to the specified direction.

        This method adds the given density data to the dictionary corresponding to the specified direction (x or y).

        :param direction: The direction to add the density data ('x' or 'y').
        :type direction: str
        :param distance: The distance at which the density is measured.
        :type distance: float
        :param density: The density value to add.
        :type density: float
        :return: None
        """
        if direction == 'x':
            self.X[distance] = density
        elif direction == 'y':
            self.Y[distance] = density