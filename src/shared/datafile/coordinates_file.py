import os
import warnings
import csv
import numpy as np
import re
from pathlib import Path

from src.shared.datafile.datafile import DataFile
from src.shared.datafile.datafile_constants import COORDINATES_EXTENSIONS
from src.shared.helpers.global_constants import DEGREE_COORDINATES_DATATYPE, EMPTY_PIXEL_COORDINATES, PIXEL_COORDINATES_DATATYPE

class CoordinatesFile(DataFile):
    """
    Class to store the csv file
    """

    def __init__(self, filename: str | None = None, datafile=None):
        allowed_extensions = COORDINATES_EXTENSIONS
        # if not filename:
        #     filename = str(datafile)
        #     if filename.endswith('.tif'):
        #         filename = re.sub('.tif', '.csv', filename)
        if filename:
            extension = filename[filename.rfind('.') + 1:]
            if extension.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {extension}")
        super().__init__(filename, datafile)

    def read_data(self, directory: Path | str) -> None:
        """
        Reads the data from a csv file and store it
        WARNING: inverts the coordinates (if x,y -> y,x)

        :param directory: string with the path to read the file
        :raise ValueError: if the extension is not supported or if the file is not found in the directory
        :raise RuntimeError: if th extension is not csv or txt
        """
        path_to_file = os.path.join(directory, self.__str__())

        if not os.path.exists(path_to_file):
            warnings.warn(
                f'File {path_to_file} does not exist, coordinates will remain empty')
            return

        if self.extension == 'csv':
            with open(path_to_file, 'r') as file:
                reader = csv.reader(file)
                coordinate_list = []
                try:
                    for row in reader:
                        coordinate_list.append(row)
                except UnicodeDecodeError:
                    raise ValueError(
                        f'{type(self).__name__}: cannot read data, because the extension '
                        f'"{self.extension}" is not supported by the csv reader or the image is not in {directory}.')

                # the list contains floats, so we cannot convert them to integers right away
                coordinates = np.array(
                    coordinate_list, dtype=DEGREE_COORDINATES_DATATYPE)

        elif self.extension == 'txt':
            coordinates = np.loadtxt(path_to_file)

        else:
            raise RuntimeError(
                f'Unsupported file extension "{self.extension}", cannot read data')

        # the coordinates are written in the format (x, y), and we need (y, x)
        if coordinates.size != 0:
            self.data = np.fliplr(np.asarray(
                np.rint(coordinates), dtype=PIXEL_COORDINATES_DATATYPE))
        else:
            self.data = EMPTY_PIXEL_COORDINATES

    def write_data(self, directory: str) -> None:
        """
        Writes the data into a csv file and save it

        :param directory: string with the path to save the file
        :raise RuntimeError: if the extension in not a csv or txt format
        """
        """
        Stores the data of the represented file in a special field, ready to be accessed.
        Coordinates in the file are stored as (x, y) pairs, but in the object as (y, x).

        WARNING: if the data is cell coordinates, they will be stored as integers!

        :param directory: directory in which the file lies (since there may be several files with the same names in
            different directories, the class itself does not store it)
        :raises ValueError: if it is impossible to read the file, either due to an incompatible format
            or to a wrong path
        """
        path_to_file = os.path.join(directory, self.__str__())
        # we write coordinate in (x, y) format to be compatible with human-labelled data
        xy_coords = np.fliplr(self.data)
        if self.extension == 'csv':
            with open(path_to_file, 'w', newline='') as file:
                writer = csv.writer(file)
                for row in xy_coords:
                    writer.writerow(row)

        elif self.extension == 'txt':
            np.savetxt(path_to_file, xy_coords)

        else:
            raise RuntimeError(
                f'Unsupported file extension "{self.extension}", cannot write data')
        
    def to_image_file(self):
        """
        Convert the current object to an ImageFile object.

        This method converts the current object to an ImageFile object by generating
        an appropriate image file name. If the current object's string representation
        does not end with '.tif', it appends '.tif' to the file name. It also removes
        any '_labeled' suffix from the file name.

        :return: An ImageFile object with the generated image file name.
        :rtype: ImageFile
        """
        from .image_file import ImageFile
        image_name = os.path.splitext(self.__str__())[0]
        if not '.tif' in self.__str__():
            image_name += '.tif'
        image_name = re.sub('_labeled', '', image_name)
        image_file = ImageFile(image_name)
        return image_file
