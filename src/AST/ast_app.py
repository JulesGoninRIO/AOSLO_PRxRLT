import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
from typing import List, Dict, Tuple

try:
    from .constants import Indices, IMAGE_DIR, AOSLO_IMAGE_SIZE_IN_DEGREES, PIXEL_COORDINATES_DTYPE, MAX_AOSLO_ECCENTRICITY, \
        OPTIC_DISC_DISTANCE, FUNDUS_IMAGE_SIZES_IN_DEGREES
    from .scaling import ScalePopup
except ImportError:
    from constants import Indices, IMAGE_DIR, AOSLO_IMAGE_SIZE_IN_DEGREES, PIXEL_COORDINATES_DTYPE, MAX_AOSLO_ECCENTRICITY, \
        OPTIC_DISC_DISTANCE, FUNDUS_IMAGE_SIZES_IN_DEGREES
    from scaling import ScalePopup


class MainMenu(tk.Menu):
    """
    Class to handle the menu in the main window to change image once we are done
    """
    def __init__(self, master):
        self.master = master
        super().__init__(master)

        self.add_command(label='Change image', command=self.master.reload_fundus_image)

class App(tk.Tk):
    """
    Class that handles the main Windows that will contain the images
    """

    def __init__(self):
        super().__init__()
        self.title('Acquisition Support Tool')
        self.protocol("WM_DELETE_WINDOW", self.quit_program)

        # the window will contain two square images and the overall width is
        # 90% of the screenwidth
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width_ratio = 0.9
        self.image_height = self.image_width = \
            int(min(screen_height, screen_width // 2) * width_ratio)

        # to the left will be the fundus image, to the right the current zooming
        # the fundus is subscribed with the position of the mouse, the zooming -
        # with position of the center of corresponding AOSLO image
        self.image_widgets: list[tk.Label] = [self._create_image_widget(ind) for ind in Indices]
        # variables used by test labels, the text is updated when the variables are updated
        self.texts: list[tk.StringVar] = [tk.StringVar(self) for _ in Indices]
        self.text_labels: list[tk.Label] = [self._create_image_label(ind) for ind in Indices]
        self.is_right_eye = True

        # creates objects to contain the images to show
        self.input_folder = IMAGE_DIR
        self.images_to_show: list[Optional[ImageTk.PhotoImage]] = [None for _ in Indices]
        self.image_shape = np.empty(2, dtype=PIXEL_COORDINATES_DTYPE)
        self.image_shape_in_degrees = np.empty(2)

        # viusal parameters and output selection
        self.grid_step = np.empty(2, dtype=PIXEL_COORDINATES_DTYPE)
        self.center_node_coordinates = np.empty(2, dtype=PIXEL_COORDINATES_DTYPE)
        self.degrees_per_pixel, self.image_center = self.load_fundus_image()
        self.marked_locations = []
        self.zoomed_location: Tuple[int, int] = None

        self.config(menu=MainMenu(self))

        self.show_images()
        self.lift()

        # select zoomed view on left mouse click
        self.bind('<Button-1>', self.select_view)
        self.bind('<Button-3>', self.mark_location)

    def quit_program(self, event=None):
        self.shutdown()

    def shutdown(self):
        self.destroy()
        # sys.exit(0)

    def _create_image_widget(self, image_ind: Indices) -> tk.Label:
        widget = tk.Label(self, bg='black')
        widget.grid(row=1, column=image_ind.value)
        return widget

    def _create_image_label(self, image_ind: Indices) -> tk.Label:
        label = tk.Label(self, bg='white', textvariable=self.texts[image_ind.value])
        label.grid(row=2, column=image_ind.value)
        return label

    def load_fundus_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes the fundus image chosen by the user, determines its size in degrees and therefore its degrees-to-pixel
        ratio.

        :return: (y, x) degrees-to-pixel ratio and (y, x) center coordinates in pixels
        """

        image_path = self.get_input_image_path()
        self.fundus_image: Image.Image = Image.open(image_path)
        self.fundus_image = self.fundus_image.resize((self.image_width, self.image_height))
        self.image_shape[0] = self.image_height
        self.image_shape[1] = self.image_width

        # get the filename
        image_path = os.path.normpath(image_path)
        filename = image_path.split(os.sep)[-1]
        label = tk.Label(self, bg='white', text=filename)
        label.grid(row=0, column=0)

        if 'OD' in filename:
            self.is_right_eye = True
        elif 'OS' in filename:
            self.is_right_eye = False
        else:
            messagebox.showerror(
                master=self,
                title='',
                message='Cannot determine if the image belongs to right or left eye.\n\n'
                        'The name of the file must contain label OS or OD.'
            )

        scale, center = self.determine_scale_and_center(filename)
        self.center_node_coordinates = center - self.grid_step // 2

        # determine the region which is visible with the AOSLO from the coordinates of top left corner
        # of the selected cell of the grid
        self.leftmost_grid = self.center_node_coordinates[1] - self.grid_step[1] * MAX_AOSLO_ECCENTRICITY
        self.rightmost_grid = self.center_node_coordinates[1] + self.grid_step[1] * (MAX_AOSLO_ECCENTRICITY + 1) + 1
        self.topmost_grid = self.center_node_coordinates[0] - self.grid_step[0] * MAX_AOSLO_ECCENTRICITY - 1
        self.bottommost_grid = self.center_node_coordinates[0] + self.grid_step[0] * (MAX_AOSLO_ECCENTRICITY + 1)

        return scale, center

    def show_fundus_image(self) -> Optional[Image.Image]:
        """
        Shows the image with superimposed grid of 1-degree stride. Highlights the current selection in red and all
        the marked locations in green.

        :return: the currently selected window, if any
        :rtype: Optional[Image.Image]
        """

        # prepare the parameters for the window
        fundus_ind = Indices.FUNDUS_IND.value
        window_size_in_pixels = np.asarray(
            np.rint(AOSLO_IMAGE_SIZE_IN_DEGREES / self.degrees_per_pixel), \
            dtype=PIXEL_COORDINATES_DTYPE
        )

        modified_fundus = self.fundus_image.copy()
        result = None
        # highlight the current zooming
        if self.zoomed_location is not None:
            up, left = self.zoomed_location
            bottom, right = self.zoomed_location + window_size_in_pixels

            window = self.fundus_image.crop((left, up, right, bottom))
            highlighter = Image.new(window.mode, window.size, color=self.highlight_color)
            modified_fundus.paste(Image.blend(window, highlighter, 0.2), (left, up, right, bottom))
            result = window

        # highlight all the user markings
        for marked_location in self.marked_locations:
            up, left = marked_location
            bottom, right = np.array(marked_location, dtype=PIXEL_COORDINATES_DTYPE) + window_size_in_pixels

            window = modified_fundus.crop((left, up, right, bottom))
            highlighter = Image.new(window.mode, window.size, color='green')
            modified_fundus.paste(Image.blend(window, highlighter, 0.4), (left, up, right, bottom))

        grid_image = np.array(modified_fundus)

        x_idx = np.arange(self.leftmost_grid, self.rightmost_grid, self.grid_step[1])
        x_idx = np.clip(x_idx, 0, grid_image.shape[1] - 1)
        y_idx = np.arange(self.topmost_grid, self.bottommost_grid, self.grid_step[0])
        y_idx = np.clip(y_idx, 0, grid_image.shape[0] - 1)

        # superimpose the grid within AOSLO eccentricities
        grid_image[max(self.topmost_grid, 0): min(self.bottommost_grid, self.image_shape[0]), x_idx] = 255
        grid_image[y_idx, max(self.leftmost_grid, 0): min(self.rightmost_grid, self.image_shape[1])] = 255

        # mark the center
        grid_image[self.image_center[0] - 1: self.image_center[0] + 2, self.image_center[1], 0] = 255
        grid_image[self.image_center[0], self.image_center[1] - 1: self.image_center[1] + 2, 0] = 255
        grid_image[self.image_center[0] - 1: self.image_center[0] + 2, self.image_center[1], 1:3] = 0
        grid_image[self.image_center[0], self.image_center[1] - 1: self.image_center[1] + 2, 1:3] = 0
        # take care of the alpha-channel
        if grid_image.shape[2] == 4:
            grid_image[self.image_center[0] - 1: self.image_center[0] + 2, self.image_center[1], 3] = 255
            grid_image[self.image_center[0], self.image_center[1] - 1: self.image_center[1] + 2, 3] = 255

        self.images_to_show[fundus_ind] = ImageTk.PhotoImage(image=Image.fromarray(grid_image), master = self)
        self.image_widgets[fundus_ind].configure(image=self.images_to_show[fundus_ind])

        return result

    def show_zoomed_image(self):
        """
        Show the zoomed image on the right part of the window
        """
        zoomed_ind = Indices.ZOOMED_IND.value
        self.image_widgets[zoomed_ind].configure(image=self.images_to_show[zoomed_ind])

    def show_images(self) -> None:
        """
        Show the image on the left of the screen and the zoomed version on the
        right
        """
        self.show_fundus_image()
        self.show_zoomed_image()

    def get_input_image_path(self) -> str:
        """
        Ask the user to choose a fundus image from the disc.

        :return: full path to the image file
        :rtype: str
        """

        image_path = filedialog.askopenfilename(
            parent=self,
            initialdir=self.input_folder,
            title='Please, choose the fundus image'
        )

        return image_path

    def _convert_mouse_coordinates_to_image_pixels(self, event: tk.Event) -> np.array:
        """
        Convert the mouse coordinates into image pixels

        :param event: the event with the mouse coordinates to disply the zoomed version
        :type event: tk.Event
        :return: (y, x) pixel coordinates of the mouse
        :rtype: np.array
        """

        # check that the mouse click is within the fundus image
        if event.widget is not self.image_widgets[0]:
            return None

        widget = self.image_widgets[0]
        x = widget.winfo_pointerx() - widget.winfo_rootx()
        y = widget.winfo_pointery() - widget.winfo_rooty()

        # widgets have borders, so coordinates in them do not precisely match coordinates on images
        pad_x = self.image_widgets[0].winfo_width() - self.image_width
        pad_y = self.image_widgets[0].winfo_height() - self.image_height
        x -= pad_x // 2
        y -= pad_y // 2

        # clip to the size of the images
        x = max(min(x, self.image_shape[0] - 1), 0)
        y = max(min(y, self.image_shape[1] - 1), 0)

        return np.array([y, x], dtype=PIXEL_COORDINATES_DTYPE)

    def determine_scale_and_center(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the degree-to-pixel ration for this image and the coordinates of its center.

        :param filename: path to the image (actually, only the filename itself is needed)
        :type filename: str
        :return: (y, x) scale and (y, x) center position
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        # check if we can determine scale from it: HRA - 30x30 degrees, OCT - 3x3 or 6x6 mm
        if 'HRA' in filename:
            size_in_degrees = FUNDUS_IMAGE_SIZES_IN_DEGREES['HRA']
        elif 'OCTA' in filename:
            if 'HD' in filename:
                size_in_degrees = FUNDUS_IMAGE_SIZES_IN_DEGREES['OCTA_HD']
            else:
                size_in_degrees = FUNDUS_IMAGE_SIZES_IN_DEGREES['OCTA']
        else:
            size_in_degrees = None

        # if size in degrees can be determined based on the type of the mage, then do it
        if size_in_degrees is not None:
            self.highlight_color = 'red'

            size_in_degrees = np.array(size_in_degrees)
            self.image_shape_in_degrees = size_in_degrees
            self.grid_step = np.asarray(np.floor(self.image_shape / size_in_degrees), dtype=PIXEL_COORDINATES_DTYPE)
            return size_in_degrees / self.image_shape, self.image_shape // 2
        # otherwise ask the user to put two marks with known coordinates
        else:
            self.highlight_color = 'blue'

            messagebox.showinfo(
                title='Cannot determine size of the image',
                message='This file is not HRA, OCTA HD or OCTA image (judging by its name). The scale of the image will '
                        'be determined manually.\n\n' \
                        'Please, click with left mouse button first on the center of the eye and then on the center of '
                        'the optic disc.\n\n'
                        f'Please, note, then we consider the distance between them to be {OPTIC_DISC_DISTANCE :.01f} '
                        'degrees. If the actual distance is different, then the grid will be placed incorrectly and '
                        'the shown zoomings might not be the actual representation of the AOSLO views.'
            )

            # show fundus image without any grid or markings
            self.optic_disc_pos = None
            win = ScalePopup(master=self,
                                    fundus_image=ImageTk.PhotoImage(image=self.fundus_image, master = self),
                                    first_callback=self.set_center_position,
                                    second_callback=self.set_optic_disc_position)
            try:
                win.mainloop()
            except SystemExit:
                pass

            dist = np.sqrt(np.sum((self.image_center - self.optic_disc_pos) ** 2))

            degrees_per_pixel = OPTIC_DISC_DISTANCE / dist
            px_per_degree = int(round(dist / OPTIC_DISC_DISTANCE))
            self.grid_step = np.array([px_per_degree, px_per_degree])

            return np.array([degrees_per_pixel, degrees_per_pixel]), self.image_center

    def set_center_position(self, coord: np.ndarray):
        """
        Set the center positon of the image from mouse position

        :param coord: coordinate (y,x) of the mouse
        :type coord: np.ndarray
        """
        self.image_center = coord

    def set_optic_disc_position(self, coord: np.ndarray):
        """
        Set the optic disk position from mouse position

        :param coord: coordinate (y,x) of the mouse
        :type coord: np.ndarray
        """
        self.optic_disc_pos = coord

    def _get_view_location(self, mouse_coordinates_in_pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the view of the location in the zoomed grid of the window

        :param mouse_coordinates_in_pixels: (y, x) chosen coordinates in pixels
        :type mouse_coordinates_in_pixels: np.ndarray
        :return: (y, x) coordinates of the top left corner of the AOSLO image that includes the given coordinates and
            (y, x) coordinates of its center in degrees (as appear on the fundus image, x-coordinate will be inverted
            in the AOSLO)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        mouse_coordinates_in_degrees = (mouse_coordinates_in_pixels - self.image_center) * self.degrees_per_pixel
        self.display_mouse_coordinates_in_degrees(mouse_coordinates_in_degrees)

        window_coordinates_in_pixels, shift_in_degrees = self.get_closest_node_grid(mouse_coordinates_in_pixels)

        # since the AOSLO images have size more than 1 degree, the top left corner of zoomed image
        # may not be exactly a node of a grid
        delta_in_degrees = (AOSLO_IMAGE_SIZE_IN_DEGREES - 1.) / 2
        delta_in_pixels = np.asarray(np.rint(delta_in_degrees / self.degrees_per_pixel), dtype=PIXEL_COORDINATES_DTYPE)
        window_coordinates_in_pixels -= delta_in_pixels

        return window_coordinates_in_pixels, shift_in_degrees

    def select_view(self, event: tk.Event):
        """
        Set the zoomed image to the window selected by the mouse click.

        :param event: Event corresponding to a mouse click
        :type event: tk.Event
        """

        mouse_coordinates_in_pixels = self._convert_mouse_coordinates_to_image_pixels(event)
        if np.any(mouse_coordinates_in_pixels is None):
            return

        self.zoomed_location, shift_in_degrees = self._get_view_location(mouse_coordinates_in_pixels)
        # print(f'{mouse_coordinates_in_pixels = }, {self.zoomed_location = }')
        # updates the fundus image highlighting the current selection
        window = self.show_fundus_image()

        window = window.resize(size=(self.image_shape[1], self.image_shape[0]))
        # we transpose the zoomed image, since it's what it looks like in the AOSLO
        self.images_to_show[Indices.ZOOMED_IND.value] = ImageTk.PhotoImage(window.transpose(Image.FLIP_LEFT_RIGHT), master = self)
        self.show_zoomed_image()

        # reverse the X-coordinate since it is reversed in the AOSLO device
        shift_in_degrees[1] = -shift_in_degrees[1]

        y_label = 'I' if shift_in_degrees[0] > 0 else 'S'
        if self.is_right_eye:
            x_label = 'T' if shift_in_degrees[1] > 0 else 'N'
        else:
            x_label = 'N' if shift_in_degrees[1] > 0 else 'T'

        self.texts[Indices.ZOOMED_IND.value].set(f'{x_label} = {abs(shift_in_degrees[1]) :01d}, {y_label} = {abs(shift_in_degrees[0]) :01d}')

    def display_mouse_coordinates_in_degrees(self, coord: np.ndarray):
        """
        Displays the mouse coordinates in degrees on the window

        :param coord: the mouse coordinates
        :type coord: np.ndarray
        """

        self.texts[Indices.FUNDUS_IND.value].set(f'{np.flip(coord)}')

    def get_closest_node_grid(self, mouse_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the coordinates of the top left corner of the cell containing the mouse click.

        :param mouse_coordinates: (y, x) coordinates of the mouse click in pixels
        :type mouse_coordinates: np.ndarray
        :return: tuple with (y, x) coordinates of the cell in pixels and its (y, x) coordinates in degrees
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        distance_to_central_node = np.asarray(mouse_coordinates - self.image_center, dtype=float)
        shift_in_nodes = np.asarray(np.rint(distance_to_central_node / self.grid_step), dtype=PIXEL_COORDINATES_DTYPE)

        return self.center_node_coordinates + self.grid_step * shift_in_nodes, shift_in_nodes

    def mark_location(self, event: tk.Event):
        """
        On right mouse button click, a cell will be marked as visited or unmarked, if it already is.

        :param event: Right click Event with position of the mouse
        :type event: tk.Event
        """

        mouse_coordinates_in_pixels = self._convert_mouse_coordinates_to_image_pixels(event)
        if np.any(mouse_coordinates_in_pixels is None):
            return

        # get the location of the corresponding AOSLO image
        window_coordinates_in_pixels, _ = self._get_view_location(mouse_coordinates_in_pixels)
        temp = window_coordinates_in_pixels.tolist()
        if temp in self.marked_locations:
            self.marked_locations.remove(temp)
        else:
            self.marked_locations.append(temp)

        # update the fundus image highlighting or dehighlighting the selection
        self.show_fundus_image()

    def reload_fundus_image(self):
        """
        Reload images if we select to change them. Will display both original (left)
        and zoomed (right) version of the image
        """
        self.image_widgets: list[tk.Label] = [self._create_image_widget(ind) for ind in Indices]
        self.texts: list[tk.StringVar] = [tk.StringVar('') for _ in Indices]
        self.text_labels: list[tk.Label] = [self._create_image_label(ind) for ind in Indices]

        self.images_to_show: list[Optional[ImageTk.PhotoImage]] = [None for _ in Indices]

        self.marked_locations = []
        self.zoomed_location: Tuple[int, int] = None

        self.degrees_per_pixel, self.image_center = self.load_fundus_image()
        self.show_images()
