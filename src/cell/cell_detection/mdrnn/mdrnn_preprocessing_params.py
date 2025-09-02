from dataclasses import dataclass

@dataclass
class MDRNNPreProcessingParams:
    """
    Parameters for MDRNN pre-processing.

    This dataclass holds the parameters used for pre-processing in the MDRNN algorithm.

    :param method: The pre-processing method to use.
    :type method: str
    :param replace: The replacement strategy to use.
    :type replace: str
    :param range_method: The method to use for range calculation.
    :type range_method: str
    :param enhancement: The enhancement technique to apply.
    :type enhancement: str
    :param correct: The correction method to use.
    :type correct: str
    """
    method: str
    replace: str
    range_method: str
    enhancement: str
    correct: str

    def __str__(self) -> str:
        """
        Return a string representation of the pre-processing parameters.

        This method returns a string that concatenates the non-None attributes of the
        pre-processing parameters, separated by underscores.

        :return: A string representation of the pre-processing parameters.
        :rtype: str
        """
        attributes = [self.method, self.range_method, self.replace, self.correct, self.enhancement]
        filtered_attributes = [attr for attr in attributes if attr is not None]
        return "_".join(filtered_attributes)

#TODO: add folder name from those parameters
#TODO: add that if method == "zero" -> range_method = None
#TODO: add build_combination -> to build all possible combinations of parameters
    # WARNING: the combination need to be ordered so that all enhancements are at the end