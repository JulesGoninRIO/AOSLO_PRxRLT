import cv2
import os
import numpy as np
import re
import logging
from PIL import Image
from pathlib import Path
from src.shared.datafile.datafile import DataFile
from src.shared.datafile.datafile_constants import ImageModalities
from src.shared.datafile.datafile_constants import IMAGE_EXTENSIONS
from src.shared.datafile.dark_region_strategy import DarkRegionsStrategy, DarkRegionsThresholdingStrategy
from src.shared.datafile.coordinates_file import CoordinatesFile

class ImageFile(DataFile):
    """
    Class to store the images.
    """

    def __init__(self, filename: str | None = None, datafile=None):
        """
        Initialize an ImageFile instance.

        This constructor initializes an ImageFile instance by checking if the file extension
        is allowed and then calling the superclass constructor. It also initializes the
        '_dark_regions' attribute to None.

        :param filename: The name of the image file.
        :type filename: str
        :param datafile: An optional datafile to initialize from.
        :type datafile: DataFile, optional
        :raise ValueError: If the file extension is not supported.
        """
        allowed_extensions = IMAGE_EXTENSIONS
        if filename:
            extension = filename[filename.rfind('.') + 1:]
            if extension.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {extension}")
        super().__init__(filename, datafile)
        self._dark_regions = None

    def read_data(self, directory: Path | str) -> None:
        """
        Read the image and store the pixel values in the class' data attribute.

        This method reads the image from the specified directory and stores the pixel values
        in the 'data' attribute of the class. If the image is not found in the directory or
        the extension is not supported, it raises a ValueError.

        :param directory: The folder where the image is located.
        :type directory: str
        :raise ValueError: If the image is not in the folder or the extension is wrong.
        """
        name = self.__str__()
        if not name in os.listdir(directory):
            name = re.sub(r'1\.0', '1', name)
            name = re.sub(r'2\.0', '2', name)
        # self.data = cv2.imread(os.path.join(
        #     directory, name), cv2.IMREAD_GRAYSCALE)
        self.data = cv2.imread(os.path.join(
            directory, name), cv2.IMREAD_UNCHANGED)
        if self.data is None or self.data.size == 0:
            logging.info(os.path.join(directory, self.__str__()))
            raise ValueError(
                f'{type(self).__name__}: cannot read data, because the extension "{self.extension}" '
                f'is not supported by OpenCV or the image is not in {directory}.'
            )

    def erase_data(self) -> None:
        """
        Erase the image data from memory
        """
        self.data = np.empty((0, 0))

    def write_data(self, directory: str) -> None:
        """
        Write the image into a directory

        :param directory: the directory to save the image
        :raise ValueError: if the image has not the image format
        """
        if len(self.data.shape) == 2:
            # we write grayscale images as they are
            cv2.imwrite(os.path.join(directory, self.__str__()), self.data)
        elif len(self.data.shape) == 3:
            # opencv writes image with BGR channel order instead of RGB,
            # so we need to revert the image
            cv2.imwrite(os.path.join(directory, self.__str__()),
                        np.flip(self.data, axis=2))
        else:
            raise ValueError('The data is not an image')

    def to_pil_rgb_image(self) -> Image:
        """
        Convert the image data to a PIL RGB image.

        This method converts the image data stored in the instance to a PIL image in RGB format.
        If the image data is empty, it raises a ValueError.

        :return: The PIL image in RGB format.
        :rtype: Image
        :raise ValueError: If the image data is empty.
        """
        if self.data.size == 0:
            raise ValueError('The image data is empty')
        # return Image.fromarray(cv2.cvtColor(self.data, cv2.COLOR_GRAY2RGB))
        return Image.fromarray(self.data).convert('RGB')

    def to_coordinates_file(self, labeled=True) -> CoordinatesFile:
        """
        Convert the image file to a coordinates file.

        This method creates a CoordinatesFile instance from the image file. If the 'labeled'
        parameter is True, it appends '_labeled' to the filename.

        :param labeled: Whether to label the coordinates file.
        :type labeled: bool
        :return: The created CoordinatesFile instance.
        :rtype: CoordinatesFile
        """
        if labeled:
            csv_file = CoordinatesFile(self.__str__() + '_labeled.csv')
        else:
            csv_file = CoordinatesFile(self.__str__() + '.csv')
        return csv_file

    @property
    def dark_regions(self) -> np.ndarray:
        """
        Property that returns dark regions.

        This property returns the dark regions of the image. If the '_dark_regions' attribute is None,
        it builds the dark regions using the provided strategy.

        :return: The dark regions of the image.
        :rtype: np.ndarray

        :raise ValueError: If the modality is not CO.
        """
        if self.modality != ImageModalities.CO.value:
            raise ValueError('Dark regions can only be computed for CO images.')
        if self._dark_regions is None:
            self._build_dark_regions(DarkRegionsThresholdingStrategy())
        return self._dark_regions

    @dark_regions.setter
    def dark_regions(self, value: np.ndarray) -> None:
        """
        Set the dark regions of the image.

        This setter method sets the '_dark_regions' attribute to the provided value.

        :param value: The dark regions to set.
        :type value: np.ndarray
        """
        self._dark_regions = value

    def _build_dark_regions(self, strategy: DarkRegionsStrategy = None) -> None:
        """
        Build the dark regions of the image using the provided strategy.

        This method builds the dark regions of the image using the specified strategy. If no strategy
        is provided, it uses the default DarkRegionsThresholdingStrategy.

        :param strategy: The strategy to use for finding dark regions.
        :type strategy: DarkRegionsStrategy, optional
        """
        if strategy is not None:
            self._dark_regions = strategy.find_dark_regions(self.data)

    def get_frame_ind(self) -> int:
        if 'frame' in self.prefix:
            i = self.prefix.find('frame') + len('frame')
            return int(''.join(c for c in self.prefix[i:] if c.isdigit()))
            # return int(self.prefix[i:])
        else:
            try:
                # return int(self.postfix)
                return int(''.join(c for c in self.postfix if c.isdigit()))
            except ValueError:
                raise RuntimeError(f'Cannot extract frame index from {self.__str__()}')

