import sys
from typing import Callable

import tkinter as tk
from tkinter.commondialog import Dialog
from PIL import ImageTk
import numpy as np

try:
    from .constants import PIXEL_COORDINATES_DTYPE
except ImportError:
    from constants import PIXEL_COORDINATES_DTYPE

StoreCoordinatesCallback = Callable[[np.ndarray], None]


class ScalePopup(tk.Toplevel):
    """
    Class to handle the creation of the zoomed image in the Window
    """

    def __init__(
            self,
            master: tk.Tk,
            fundus_image: ImageTk.PhotoImage,
            first_callback: StoreCoordinatesCallback,
            second_callback: StoreCoordinatesCallback
    ) -> None:
        super().__init__(master=master)

        self.fundus_image = fundus_image
        self.widget = tk.Label(master=self, image=self.fundus_image)
        self.widget.grid(row=0, column=0)

        self.first_callback = first_callback
        self.second_callback = second_callback

        self.marked = 0

        self.bind('<Button-1>', self.save_coordinates)

    def save_coordinates(self, event: tk.Event):
        """
        Save coordinates in the zoomed image

        :param event: Event corresponding to a mouse click
        :type event: tk.Event
        """
        y, x = self._convert_mouse_coordinates_to_image_pixels()
        if self.marked == 0:
            self.first_callback(np.array([y, x]))
        elif self.marked == 1:
            self.second_callback(np.array([y, x]))
            self.destroy()
            sys.exit(0)

        self.marked += 1

    def _convert_mouse_coordinates_to_image_pixels(self) -> np.array:
        """
        Convert mouse coordinates to image pixels

        :return: (y, x) pixel coordinates of the mouse
        :rtype: np.array
        """
        # check that the mouse click is within the fundus image
        x = self.winfo_pointerx() - self.winfo_rootx()
        if x > self.widget.winfo_width():
            return None
        y = self.winfo_pointery() - self.winfo_rooty()
        if y > self.widget.winfo_height():
            return None

        # widgets have borders, so coordinates in them do not precisely match coordinates on images
        pad_x = self.widget.winfo_width() - self.fundus_image.width()
        pad_y = self.widget.winfo_height() - self.fundus_image.height()
        x -= pad_x // 2
        y -= pad_y // 2

        # clip to the size of the images
        x = max(min(x, self.fundus_image.width() - 1), 0)
        y = max(min(y, self.fundus_image.height() - 1), 0)

        return np.array([y, x], dtype=PIXEL_COORDINATES_DTYPE)
