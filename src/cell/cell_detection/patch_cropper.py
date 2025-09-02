from typing import List, Tuple, Dict
import math
import numpy as np

from src.shared.datafile.image_file import ImageFile

# Default value maintained for backward compatibility
DEFAULT_PATCH_SIZE = 185
MIN_OVERLAP = 14

class PatchCropper:
    """
    A class used to crop patches from an image.

    This class provides methods to generate patches from an image and retrieve their names and data.
    """
    def __init__(self, image: ImageFile = np.zeros((720, 720)), patch_size: int = DEFAULT_PATCH_SIZE):
        """
        Initialize the PatchCropper with an image.

        :param image: The image to crop patches from.
        :type image: ImageFile
        :param patch_size: Size of the square patches to be cropped.
        :type patch_size: int
        :raises ValueError: If the image dimensions are less than or equal to the patch size.
        """
        # if not isinstance(image, ImageFile):
        #     raise ValueError("Image must be an instance of ImageFile")
        self.image = image
        self.patch_size = patch_size
        self.width = self.image.data.shape[1]
        self.height = self.image.data.shape[0]
        if self.width < self.patch_size or self.height < self.patch_size:
            raise ValueError("Image dimensions must be greater than the patch size")
        if self.width == self.patch_size and self.height == self.patch_size:
            raise ValueError("Image dimensions must be greater than the patch size")

    def get_number_of_patches(self) -> Tuple[int, int]:
        """
        Get the number of patches in the x and y dimensions.
        Ensures enough patches to have at least MIN_OVERLAP overlap.

        :return: A tuple containing the number of patches in the x and y dimensions.
        :rtype: Tuple[int, int]
        """
        if self.width <= self.patch_size:
            nb_patches_x = 1
        else:
            # Calculate how many patches needed to ensure MIN_OVERLAP
            step_size = self.patch_size - MIN_OVERLAP
            nb_patches_x = max(2, math.ceil((self.width - self.patch_size) / step_size) + 1)
        
        if self.height <= self.patch_size:
            nb_patches_y = 1
        else:
            step_size = self.patch_size - MIN_OVERLAP
            nb_patches_y = max(2, math.ceil((self.height - self.patch_size) / step_size) + 1)
            
        return nb_patches_x, nb_patches_y

    def find_patches_positions(self, nb_patches: int, dimension_size: int) -> np.ndarray:
        """
        Find the positions of patches along a dimension.
        Ensures minimum overlap between patches.

        :param nb_patches: The number of patches.
        :type nb_patches: int
        :param dimension_size: The size of the dimension.
        :type dimension_size: int
        :return: An array of patch positions.
        :rtype: np.ndarray
        """
        if nb_patches == 1:
            return np.array([0])
        
        # Calculate positions ensuring minimum overlap
        total_distance = dimension_size - self.patch_size
        positions = []
        
        if nb_patches > 2:
            step = total_distance / (nb_patches - 1)
            if step > (self.patch_size - MIN_OVERLAP):
                # Recalculate with enforced overlap
                step = self.patch_size - MIN_OVERLAP
                nb_patches = max(2, math.ceil(total_distance / step) + 1)
                
            positions = [i * step for i in range(nb_patches)]
        else:
            # Just two patches - one at start, one at end
            positions = [0, total_distance]
            
        # Verify minimum overlap is maintained
        for i in range(len(positions)-1):
            overlap = self.patch_size - (positions[i+1] - positions[i])
            if overlap < MIN_OVERLAP:
                raise ValueError(f"Could not maintain minimum overlap of {MIN_OVERLAP} pixels. Got {overlap} pixels.")
                
        return np.array(positions)

    def __generate_patch_info(self, counter: int = 0) -> Tuple[List[tuple], int]:
        """
        Generate information about the patches.

        :param counter: The starting counter value.
        :type counter: int
        :return: A tuple containing a list of patch information and the updated counter.
        :rtype: Tuple[List[tuple], int]
        """
        nb_patches_x, nb_patches_y = self.get_number_of_patches()
        x_positions = self.find_patches_positions(nb_patches_x, self.width)
        y_positions = self.find_patches_positions(nb_patches_y, self.height)

        patch_info = []
        for i in range(len(x_positions)):
            for j in range(len(y_positions)):
                counter += 1
                x_pos = math.floor(x_positions[i])
                y_pos = math.floor(y_positions[j])
                patch_info.append((counter, x_pos, y_pos))
        return patch_info, counter

    def get_patches_names(self) -> List[str]:
        """
        Get the names of the patches.

        :return: A list of patch names.
        :rtype: List[str]
        """
        patch_names = []
        for counter, x_pos, y_pos in self.__generate_patch_info():
            patch_name = f"CROP_{counter}_x{x_pos}y{y_pos}_" + str(self.image)
            patch_names.append(patch_name)
        return patch_names

    def get_patches(self, counter) -> Tuple[List[ImageFile],int]:
        """
        Get the patches from the image.

        :param counter: The starting counter value.
        :type counter: int
        :return: A list of ImageFile objects representing the patches and the updated counter.
        :rtype: List[ImageFile], int
        """
        patches = []
        patch_info_list, counter = self.__generate_patch_info(counter)
        for patch_info in patch_info_list:
            counter, x_pos, y_pos = patch_info
            patch_name = f"CROP_{counter}_x{x_pos}y{y_pos}_" + str(self.image)
            new_patch = ImageFile(patch_name)
            new_patch.data = self.image.data[x_pos:x_pos+self.patch_size, y_pos:y_pos+self.patch_size]
            patches.append(new_patch)
        return patches, counter

    def get_overlapping_regions(self) -> List[Tuple[int, int, int, int]]:
        """
        Get overlapping regions between patches.

        This method calculates the overlapping regions between patches in the x and y dimensions.

        :return: A list of tuples representing the overlapping regions. Each tuple contains the start and end coordinates of the overlapping region.
        :rtype: List[Tuple[int, int, int, int]]
        """
        overlapping_regions = []
        nb_patches_x, nb_patches_y = self.get_number_of_patches()
        x_positions = self.find_patches_positions(nb_patches_x, self.width)
        y_positions = self.find_patches_positions(nb_patches_y, self.height)

        # Iterate through each pair of patches
        for i in range(nb_patches_x):
            for j in range(nb_patches_y):
                x_pos_i = math.floor(x_positions[i])
                y_pos_j = math.floor(y_positions[j])

                for k in range(i + 1, nb_patches_x):  # Ensure each pair is only checked once
                    x_pos_k = math.floor(x_positions[k])
                    x_overlap_start = max(x_pos_i, x_pos_k)
                    x_overlap_end = min(x_pos_i + self.patch_size, x_pos_k + self.patch_size)
                    if x_overlap_end > x_overlap_start:
                        overlapping_regions.append((x_overlap_start, y_pos_j, x_overlap_end, y_pos_j + self.patch_size))
                for l in range(j + 1, nb_patches_y):
                    y_pos_l = math.floor(y_positions[l])
                    y_overlap_start = max(y_pos_j, y_pos_l)
                    y_overlap_end = min(y_pos_j + self.patch_size, y_pos_l + self.patch_size)
                    if y_overlap_end > y_overlap_start:
                        overlapping_regions.append((x_pos_i, y_overlap_start, x_pos_i + self.patch_size, y_overlap_end))

        return overlapping_regions
    

# if __name__ == "__main__":
#     import sys
#     # sys.path.append(r"C:\Users\BardetJ\Downloads\aoslo_pipeline-master")
#     import cv2
#     overlap_regions = PatchCropper().get_overlapping_regions()
#     image = np.ones((720, 720, 3), dtype=np.uint8) * 255
#     for (x1, y1, x2, y2) in overlap_regions:
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.imwrite(r'C:\Users\BardetJ\Downloads\cones.tif', (image).astype(np.uint8))