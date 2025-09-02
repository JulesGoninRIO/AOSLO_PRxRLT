import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional
import cv2
import csv
import logging

import numpy as np
from PIL import Image, ImageTk
import PIL
from pathlib import Path
import re
from scipy import io
from tkinter.ttk import Style, Button
import screeninfo

try:
    from .constants import PIXEL_COORDINATES_DTYPE, \
        OPTIC_DISC_DISTANCE, FUNDUS_IMAGE_SIZES_IN_DEGREES
    from .drag_drop import DragManager
    from .helpers import find_matched_chains, find_chain_centers
except ImportError:
    from constants import PIXEL_COORDINATES_DTYPE, \
        OPTIC_DISC_DISTANCE, FUNDUS_IMAGE_SIZES_IN_DEGREES
    from drag_drop import DragManager
    from helpers import find_matched_chains, find_chain_centers

from typing import List, Dict, Tuple

import time
import math

# Global value to get whether components have been validated or suppress or to
# do later
valid = True

class App(tk.Tk):
    """
    Main GUI Window to place the unconnected components on the fundus image

    :param tk: the Tkinter window
    :type tk: tk.Tk
    """

    def __init__(self, directory: str = None):
        """
        Initialize the GUI Window with optional directory where to look for files

        :param directory: directory where the images are if any, defaults to None
        :type directory: str, optional
        """

        super().__init__()
        self.title('Acquisition Support Tool V2: Montaging Correction')
        self.protocol("WM_DELETE_WINDOW", self.__quit_program)

        # The window is separated in two for placing components on the left and
        # having the compoenents informations and useful buttons on the right
        self.frame_left = tk.Frame(self)
        self.frame_left.pack(side=tk.LEFT, expand=False, fill=tk.BOTH)
        self.frame_right = tk.Frame(self)
        self.frame_right.pack(expand=True, fill=tk.X)

        # We also need a transparent frame for the components to be loaded and
        # placed while seeing the fundus image
        self.frame_transparancy = tk.Frame(self)
        self.frame_transparancy.pack(expand=True, fill=tk.X)

        # Objects where results will be stored
        self.draw_center = None
        self.label_center = None
        self.center_loc = None

        # Initialize style of the buttons, screen size and center location button
        # where right click will mark the center of the eye
        self.__init_style()
        screen_width, screen_height = self.__monitor()
        self.bind('<Button-3>', self.__mark_location)

        # Set up base parameters for component transparancy, the transparancy
        # when we move the component and the zoomed size of the first component
        # to place
        self.transparancy = 0.7
        self.dragged_transparancy = 0.75
        self.size_constant = 1.5
        self.zoom_allowed = True

        # The window will contain two square images and the overall width is
        # 92% of the screenwidth
        width_ratio = 0.92
        self.image_height = self.image_width = int(min(screen_height, \
            screen_width) * width_ratio)

        # The components placed will be contained in this list
        self.components = []

        # The left will be the fundus image_ the fundus is subscribed with the
        # position of the mouse, the zooming - with position of the center of
        # corresponding AOSLO image
        self.image_widgets: tk.Canvas = self.__create_image_widget()
        self.text_labels: tk.Label = self.__create_image_label()

        # Initiate the base folder to the path of the current file and the fundus
        # image object
        self.input_folder = directory if directory else os.path.abspath(__file__)
        self.image_to_show: ImageTk.PhotoImage = None
        self.image_shape = np.empty(2, dtype=PIXEL_COORDINATES_DTYPE)
        self.image_shape_in_degrees = np.empty(2)

        # Load the fundus and the montage image
        self.degrees_per_pixel, self.image_center = self.__load_fundus_image()
        self.connected_components = []
        self.__load_montage_images()

        # Start the gui main loop by setting up the menu and showing components
        # one by one
        self.__show_images()

    def __init_style(self):
        """
        Creates a style object for buttons
        """

        style = Style()
        # This will be adding style, and naming that style variable as
        style.configure('W.TButton', font = ('calibri', 20, 'bold', 'underline'),
                        foreground = 'black', background = 'blue', master = self)

    def __monitor(self) -> Tuple[int, int]:
        """
        Get the informations of the screen to recover the perfect shifts once
        finished

        :return: width and height of the screen
        :rtype: Tuple[int, int]
        """

        monitors = screeninfo.get_monitors()
        if len(monitors)>1:
            logging.info("You have mutliple screens: will not use the principal monitor "\
                  "as for image size but the additional screen.")
        logging.info("Please open the window on the additional screen in full size")
        monitor = monitors[-1]
        height = monitor.height
        width = monitor.width

        return width, height

    def __quit_program(self, save: bool = False):
        """
        Quit the program and save results if needed

        :param save: whether or not we save the results, defaults to False
        :type save: bool, optional
        """

        if save:
            with open(os.path.join(self.montage_path, "locations.csv"), 'w') as f:
                write = csv.writer(f)
                write.writerow([self.center_loc, [self.image_width, self.image_height]])
                write.writerows(self.components)
        self.__shutdown(save)

    def __shutdown(self, save: bool = False):
        """
        Shutdown the GUI

        :param save: whether or not we save the results, defaults to False
        :type save: bool, optional
        """

        self.destroy()
        sys.exit(0)

    def __create_image_widget(self) -> tk.Canvas:
        """
        Creates an image widget where the fundus image will be shown

        :return: the fundus image widget
        :rtype: tk.Canvas
        """

        # Set the widget size to half of the width of the screen
        widget = tk.Canvas(self, height=self.image_height, width=self.image_width)
        widget.pack(fill=tk.BOTH, expand=True)
        return widget

    def __create_image_label(self) -> tk.Label:
        """
        Creates an image label for the fundus image on the right frame

        :return: the label of the image
        :rtype: tk.Label
        """

        label = tk.Label(self, textvariable='Fundus image')
        label.pack(in_=self.frame_right, padx=20, pady=20)

        return label

    def __load_image(self, name: str, mirror = False) -> Tuple[np.ndarray, str]:
        """
        Load an image from a directory ansd resize it to the proper window size

        :param name: the name of the image type we want (e.g. fundus)
        :type name: str
        :param mirror: whether we need to flip the fundus image, defaults to False
        :type mirror: bool, optional
        :return: the image loaded and its name
        :rtype: Tuple[np.ndarray, str]
        """

        # Get the image path, read it and resize it
        image_path = self.__get_input_image_path(name)
        try:
            image: Image.Image = Image.open(image_path)
        except PIL.UnidentifiedImageError:
            image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image: Image.Image = Image.fromarray(np.uint8(image))
        except AttributeError:
            # no file chosen
            self.__shutdown()

        image = image.resize((self.image_width, self.image_height))
        if mirror:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        self.image_shape[0] = self.image_height
        self.image_shape[1] = self.image_width

        # get the name of the loaded file
        self.image_path = os.path.normpath(image_path)
        self.input_folder = self.image_path
        filename = image_path.split(os.sep)[-1]

        return image, filename

    def __load_montage_images(self) -> None:
        """
        Takes the unconnected components from the montaged image and put them on
        the right of the fundus image
        """

        # ask the user for the montage path and the montage image with all the
        # components (for correcting Confocal images it is all_ref_combined_m1.tif)
        image_path_name = self.__get_input_image_path("montage")
        self.image_path = os.path.normpath(image_path_name)
        filename = image_path_name.split(os.sep)[-1]

        # Get all the unconnected component of the selected montage image
        self.montage_path = Path(self.image_path).parent
        modality = re.search(r"_m(\d+)", filename).group(0)
        filenames = [name for name in os.listdir(self.montage_path) if modality \
            in name and not "all" in name and "combined" in name]
        for image_name in filenames:
            self.connected_components.append(image_name)

        # look if the montaged images loaded are from the left or right eye
        filenames_left_side = [name for name in os.listdir(self.montage_path) \
            if "_OS_" in name]
        filenames_right_side = [name for name in os.listdir(self.montage_path) \
            if "_OD_" in name]
        if filenames_left_side and filenames_right_side:
            logging.info("The images are from left and right eye, cannot process them")
            self.__shutdown()

        elif filenames_left_side:
            self.is_right_eye = False
        elif filenames_right_side:
            self.is_right_eye = True
        else:
            logging.info("No image in the directory to find whether it is a left or "\
                  "right eye.")


    def __load_fundus_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes the fundus image chosen by the user, determines its size in
        degrees and therefore its degrees-to-pixel ratio.

        :return: (y, x) degrees-to-pixel ratio and (y, x) center coordinates in
                 pixels
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        # Load the fundus image by selecting it from the GUI
        self.fundus_image, filename = self.__load_image("fundus", mirror=True)
        label = tk.Label(self, text=filename)
        label.pack(in_=self.frame_right, padx=20, pady=20)

        # the side of the eye is defined in the name of the fundus file
        if 'OD' in filename:
            self.is_right_eye = True
        elif 'OS' in filename:
            self.is_right_eye = False

        # Add a label whether it is a left or a right eye
        text = ["Right Eye" if self.is_right_eye else "Left Eye"][0]
        label = tk.Label(self, text=f"{text}")
        label.pack(in_=self.frame_right, padx=20, pady=20)

        # get the center and the scale of the loaded image
        scale, center = self.__determine_scale_and_center(filename)

        return scale, center

    def __show_fundus_image(self) -> None:
        """
        Shows the image with superimposed grid of 1-degree stride. Highlights
        the current selection in red and all the marked locations in green.
        """

        # Copy the fundus image so that we can resize it without changing the
        # original shape
        modified_fundus = self.fundus_image.copy()

        # to place the first component we have a zoomed version of the fundus
        # shown at first so we need to resize it
        if self.zoom_allowed:
            size = modified_fundus.size
            new_size = self.size_constant*np.array(size)
            new_size = [round(new_size[0]), round(new_size[1])]
            modified_fundus = modified_fundus.resize(new_size)
            x_pos = (size[0]-new_size[0])/2
            y_pos = (size[1]-new_size[1])/2
        else:
            x_pos = 0
            y_pos = 0

        # create a Tkinter Image object and display it on the left frame
        self.image_to_show = ImageTk.PhotoImage(image=modified_fundus, master=self)
        self.image_widgets.create_image(x_pos, y_pos, image=self.image_to_show, \
            anchor="nw")
        self.image_widgets.pack(in_=self.frame_left)

    def __show_connected_components(self):
        """
        Shows a connected component on the left frame on top of the fundus image.
        Places buttons to manage the placement of components

        :yield: the component that is being placed
        :rtype: DragManager
        """

        global valid

        # Get the compoenents references of the components and sort them so that
        # we can load the center one at first
        connected_components_refs = [eval(re.search(r'ref_(\d+)', ref).group(1)) \
            for ref in self.connected_components]
        connected_components_refs.sort()
        self.connected_components_sorted = []
        self.component_images = []
        for connected_component_ref in connected_components_refs:
            self.connected_components_sorted.append([val for val in self.connected_components \
                if re.search(f'ref_{connected_component_ref}_', val)][0])
            self.component_images.append(None)

        # read the Matlab's montaged result data
        filename = os.path.join(self.montage_path, 'AOMontageSave.mat')
        data = io.loadmat(filename)
        # since in Matlab numeration starts from 1
        matched_to = data['MatchedTo'].flatten() - 1
        # despite the name, vertical coordinates go first
        locations = data['LocXY']
        self.locations = np.transpose(locations)
        matched_chains = find_matched_chains(matched_to)
        chain_centers = find_chain_centers(matched_chains, self.locations)
        self.matched_chains = list(matched_chains.values())
        self.chain_centers = list(chain_centers.values())

        def __ask_valid():
            """
            Button that ask the user if they valid the component location. When
            pressed, this function is run to save the current component information
            and then the next component is loaded and displayed

            :yield: the component while the button is not pressed on the GUI
            :rtype: DragManager
            """

            i=0

            # go through the components one by one by yielding the current
            # component waiting for User inputs
            for i in range(len(self.connected_components_sorted)):
                self.index = i

                label1, label2 = self.__display_component_info(i)
                self.component = self.connected_components_sorted[i]

                # if it is the first component we allow zooming
                if self.zoom_allowed:
                    self.component_images[i] = DragManager(self.montage_path, \
                        self.image_widgets, self.component, \
                        round(self.size_constant*self.image_height), \
                        round(self.size_constant*self.image_width), \
                        transparancy = self.transparancy, \
                        dragged_transparancy = self.dragged_transparancy, \
                        zoom_allowed=self.zoom_allowed)
                else:
                    self.component_images[i] = DragManager(self.montage_path, \
                        self.image_widgets, self.component, \
                        self.image_height, self.image_width, \
                        transparancy = self.transparancy, \
                        dragged_transparancy = self.dragged_transparancy, \
                        zoom_allowed=self.zoom_allowed)

                # wait that the user place component and click on validation button
                yield self.component_images[i]

                if self.zoom_allowed and valid:
                    # as it is the first component we nned to reload the unzoomed
                    # fundus image, suppress the zoomed first component and
                    # show the unzoomed version of the first component
                    # we also ask for valid to be true otherwise it means we want
                    # to do later the first component or we suppress it so we want
                    # to have the second component zoomed as well
                    xpos, ypos, size, scale, angle = self.component_images[i].\
                        get_coordinates()
                    self.zoom_allowed = False
                    self.__show_fundus_image()
                    middle = [self.image_width/2, self.image_height/2]
                    xpos -= (xpos - middle[0])/(2*self.size_constant)
                    ypos -= (ypos - middle[1])/(2*self.size_constant)
                    size = [round(size[0]/(scale*self.size_constant)), \
                        round(size[1]/(scale*self.size_constant))]
                    self.component_images[i].delete()
                    self.component_images[i] = DragManager(self.montage_path, \
                        self.image_widgets, self.component, size[1], size[0], \
                        xpos, ypos, transparancy = self.transparancy, \
                        dragged_transparancy = self.dragged_transparancy, \
                        zoom_allowed=self.zoom_allowed, angle=angle)

                if not valid:
                    # we will do the component at the end
                    self.component_images[i].delete()
                    try:
                        self.components.append([self.connected_components_sorted[i], \
                            None, None, size, i, None])
                    except UnboundLocalError:
                        # do later for the first component
                        self.components.append([self.connected_components_sorted[i], \
                            None, None, None, i, None])
                else:
                    # save the components informations
                    xpos, ypos, size, _, angle = \
                        self.component_images[i].get_coordinates()
                    self.components.append([self.connected_components_sorted[i], \
                        xpos, ypos, size, i, angle])
                label1.destroy()
                label2.destroy()

            # if we have components to do later or cancelled it will be done now
            num_misses = len(self.connected_components_sorted)-1-i
            if num_misses>0:
                # no more cancel possible once we have to redo the components
                C.destroy()
                label0 = tk.Label(self, bg='red', text=f"Redoing the cancelled "\
                    "components:\n please don't miss them, there is no cancel anymore", \
                    font=("Arial", 16))
                label0.pack(in_=self.frame_right, padx=20, pady=20)

            # do the cancelled or do later component as before
            for j in range(1,num_misses+1):
                index = self.connected_components_sorted[i+j][-1]
                label1, label2 = self.__display_component_info(index)
                if i == 0:
                    self.zoom_allowed = True
                else:
                    self.zoom_allowed = False
                self.component = self.connected_components_sorted[i+j][0]
                self.component_images[i] = DragManager(self.montage_path, \
                    self.image_widgets, self.component, self.image_height, \
                    self.image_width, transparancy = self.transparancy, \
                    dragged_transparancy = self.dragged_transparancy, \
                    zoom_allowed=self.zoom_allowed)
                yield self.component_images[i]

                # delete if not valid else save the informations
                if not valid:
                    self.component_images[i].delete()
                    if self.connected_components_sorted[i] in np.array(self.components, \
                        dtype = object)[:,0]:
                        indice = np.argwhere(np.array(self.components, dtype = object)\
                            [:,0] == self.connected_components_sorted[i])[0][0]
                        self.components.pop(indice)
                    self.components.append([self.connected_components_sorted[i+j][0], \
                        None, None, size, index, None])
                else:
                    xpos, ypos, size, _, angle = self.component_images[i].get_coordinates()
                    if self.connected_components_sorted[i] in np.array(self.components, \
                        dtype = object)[:,0]:
                        indice = np.argwhere(np.array(self.components, dtype = object)\
                            [:,0] == self.connected_components_sorted[i])[0][0]
                        self.components.pop(indice)
                    self.components.append([self.connected_components_sorted[i+j][0], \
                        xpos, ypos, size, index, angle])
                label1.destroy()
                label2.destroy()

        a = __ask_valid()
        var = tk.IntVar()

        # at first we get the fundus image so we should first get coordinates of the
        # very center and save it
        button_save_center = Button(self, text="Select center of the eye and " \
            "press when done", command = lambda: var.set(1), style = 'W.TButton')
        button_save_center.pack(in_=self.frame_right, padx=20, pady=20)
        label_right_click = tk.Label(self, bg='red', text=f"Right click on the " \
            "very center of the retina", font=("Arial", 16))
        label_right_click.pack(in_=self.frame_right, padx=20, pady=20)
        button_save_center.wait_variable(var)
        while not self.draw_center:
            label_right_click.destroy()
            label_right_click = tk.Label(self, bg='orange', text=f"You should " \
                "first pick with right click the very center of the eye before " \
                "validating it")
            label_right_click.pack(in_=self.frame_right, padx=20, pady=20)
            button_save_center.wait_variable(var)
        self.unbind('<Button-3>')
        label_right_click.destroy()

        # Once the center is selected we load the first component image by first
        # showing a loading text because it takes some time to display the image
        self.__display_loading()
        next(a)
        self.label_center.destroy()
        label_start = tk.Label(self, bg='red', text="Starting the placement of " \
            "components", font=("Arial", 16))
        label_start.pack(in_=self.frame_right, padx=20, pady=20)
        button_save_center.destroy()

        def __continue_loop_valid():
            """
            Validate the placed component and continue to the next component
            """

            global valid
            valid = True
            self.__display_loading()
            try:
                next(a)
            except StopIteration:
                self.__display_end_gui()

        def __continue_loop_suppress():
            """
            Suppress the component and continue to the next one
            """
            global valid
            valid = False
            self.__display_loading()
            try:
                next(a)
            except StopIteration:
                self.__display_end_gui()

        # creates buttons to handle the saving and suppressinf of components
        V = Button(self, text="Valid image position", command = __continue_loop_valid, \
            style = 'W.TButton')
        S = Button(text="Suppress image", command = __continue_loop_suppress, \
            style = 'W.TButton')
        V.pack(in_=self.frame_right, padx=10, pady=10)
        S.pack(in_=self.frame_right, padx=10, pady=10)

        def __cancel():
            """
            Functions that allows to cancel the last component placed which was
            either saved or suppressed
            """

            # if valid is False we reload the compoenent but not remove last on list
            # if valid is True, we should reload and remove last
            label1 = tk.Label(self, bg='red', text="The last component has been "\
                "cancelled,\n you will be ale to redo it at the end", font=("Arial", 16))
            label1.pack(in_=self.frame_right, padx=20, pady=20)
            self.after(5000, lambda: label1.destroy())
            self.__display_loading()
            global valid
            if valid:
                self.component_images[self.components[-1][-2]].delete()
            self.connected_components_sorted.append([self.components[-1][0], self.components[-1][-2]])
            self.components.pop()

        C = Button(self, text="Cancel", command = __cancel, style = 'W.TButton')
        C.pack(in_=self.frame_right, padx=10, pady=10)

        def __do_later():
            """
            Allows to do later a component if the doctor is not sure about its
            placement
            """

            # add the component at the end of the list and go to the next
            # component
            label1 = tk.Label(self, bg='red', text="You will be able to do this " \
                "component at the end", font=("Arial", 16))
            label1.pack(in_=self.frame_right, padx=20, pady=20)
            self.after(5000, lambda: label1.destroy())
            self.__display_loading()
            try:
                index = np.argwhere(np.array(self.connected_components_sorted, \
                    dtype = object) == self.components[-1][0])[0][0]+1
            except IndexError:
                # Means you want to do later the first component
                index = 0
            component = self.connected_components_sorted[index]
            self.connected_components_sorted.append([component, index])
            global valid
            valid = False
            next(a)

        D = Button(self, text="Do Later", command = __do_later, style = 'W.TButton')
        D.pack(in_=self.frame_right, padx=10, pady=10)

        def __rm_transparancy():
            """
            Allows to reduce transparancy of an image so that we see it better
            """

            # change transparancy and reload the component with updated transparancy
            self.transparancy += 0.1
            self.t.set(f"Image Transparancy \u00B10.1: {round(1-self.transparancy, 1)}")
            self.__display_loading()
            xpos, ypos, size, scale, _ = self.component_images[self.index].get_coordinates()
            self.component_images[self.index].delete()

            self.component_images[self.index] = DragManager(self.montage_path, \
                self.image_widgets, self.component, round(size[1]/scale), \
                round(size[0]/scale), xpos, ypos, transparancy = self.transparancy, \
                dragged_transparancy = self.dragged_transparancy, \
                zoom_allowed=self.zoom_allowed)

        minus1 = tk.Button(self, text="-", command = __rm_transparancy)
        minus1.pack(in_=self.frame_transparancy, padx=(100, 10), pady=10, side=tk.LEFT)

        # Write the transparancy value that is used
        self.t = tk.StringVar()
        self.t.set(f"Image Transparancy \u00B10.1: {round(1-self.transparancy, 1)}")
        label_transparancy = tk.Label(self, bg='orange', textvariable=self.t)
        label_transparancy.pack(in_=self.frame_transparancy, padx=20, pady=20, side=tk.LEFT)

        def __add_transparancy():
            """
            Allows to add transparancy of an image so that we see it less
            """

            # change transparancy and reload the component with updated transparancy
            self.transparancy -= 0.1
            self.t.set(f"Image Transparancy \u00B10.1: {round(1-self.transparancy, 1)}")
            self.__display_loading()
            xpos, ypos, size, scale, _ = self.component_images[self.index].get_coordinates()
            self.component_images[self.index].delete()
            self.component_images[self.index] = DragManager(self.montage_path, \
                self.image_widgets, self.component, round(size[1]/scale), \
                round(size[0]/scale), xpos, ypos, transparancy = self.transparancy, \
                dragged_transparancy = self.dragged_transparancy, \
                zoom_allowed=self.zoom_allowed)
            self.update()

        plus1 = tk.Button(self, text="+", command = __add_transparancy)
        plus1.pack(in_=self.frame_transparancy, padx=(10, 100), pady=10, side=tk.LEFT)

        # Transparancy of the dragged image
        def __rm_transparancy_bis():
            """
            Allows to reduce dragged transparancy of an image so that we see
            it better
            """

            # change dragged transparancy and reload the component with updated
            # dragged transparancy
            self.dragged_transparancy += 0.25
            self.t_d.set(f"Image Dragged Transparancy \u00B10.25: {round(1-self.dragged_transparancy, 2)}")
            self.__display_loading()
            xpos, ypos, size, scale, _ = self.component_images[self.index].get_coordinates()
            self.component_images[self.index].delete()
            self.component_images[self.index] = DragManager(self.montage_path, \
                self.image_widgets, self.component, round(size[1]/scale), \
                round(size[0]/scale), xpos, ypos, transparancy = self.transparancy, \
                dragged_transparancy = self.dragged_transparancy, \
                zoom_allowed=self.zoom_allowed)
            self.update()

        minus2 = tk.Button(self, text="-", command = __rm_transparancy_bis)
        minus2.pack(in_=self.frame_transparancy, padx=10, pady=10, side=tk.LEFT)

        self.t_d = tk.StringVar()
        self.t_d.set(f"Image Dragged Transparancy \u00B10.25: {round(1-self.dragged_transparancy, 2)}")
        label_transparancy_d = tk.Label(self, bg='orange', textvariable=self.t_d)
        label_transparancy_d.pack(in_=self.frame_transparancy, padx=20, pady=20, side=tk.LEFT)

        def __add_transparancy_bis():
            """
            Allows to add dragged transparancy of an image so that we see
            it less
            """

            # change dragged transparancy and reload the component with updated
            # dragged transparancy
            self.dragged_transparancy -= 0.25
            self.t_d.set(f"Image Dragged Transparancy \u00B10.25: {round(1-self.dragged_transparancy, 2)}")
            self.__display_loading()
            xpos, ypos, size, scale, _ = self.component_images[self.index].get_coordinates()
            self.component_images[self.index].delete()
            self.component_images[self.index] = DragManager(self.montage_path, \
                self.image_widgets, self.component, round(size[1]/scale), \
                round(size[0]/scale), xpos, ypos, transparancy = self.transparancy, \
                dragged_transparancy = self.dragged_transparancy, \
                zoom_allowed=self.zoom_allowed)
            self.update()

        plus2 = tk.Button(self, text="+", command = __add_transparancy_bis)
        plus2.pack(in_=self.frame_transparancy, padx=10, pady=10, side=tk.LEFT)

    def __show_images(self) -> None:
        """
        Start the GUI loop by showing first the fundus image and then every single
        component one by one
        """
        self.__show_fundus_image()
        self.__show_connected_components()

    def __get_input_image_path(self, name: str) -> str:
        """
        Ask the user to choose a fundus image from the disc.

        :param name: the name of the image we want
        :type name: str
        :return: full path to the image file
        :rtype: str
        """

        # ask to choose a fundus image
        image_path = filedialog.askopenfilename(
            parent=self,
            initialdir=self.input_folder,
            title=f'Please, choose the {name} image'
        )

        return image_path

    def __convert_mouse_coordinates_to_image_pixels(self, event: tk.Event) -> np.array:
        """
        Converts the mouse coordinates into image pixels

        :param event: position of the mouse when the event is triggered
        :type event: tk.Event
        :return: (y, x) pixel coordinates of the mouse
        :rtype: np.array
        """

        # check that the mouse click is within the fundus image
        if event.widget is not self.image_widgets:
            return None

        # get the position in pixels
        widget = self.image_widgets
        x = widget.winfo_pointerx() - widget.winfo_rootx()
        y = widget.winfo_pointery() - widget.winfo_rooty()

        # widgets have borders, so coordinates in them do not precisely match
        # coordinates on images
        pad_x = self.image_widgets.winfo_width() - self.image_width
        pad_y = self.image_widgets.winfo_height() - self.image_height
        x -= pad_x // 2
        y -= pad_y // 2

        # clip to the size of the images
        x = max(min(x, self.image_shape[0] - 1), 0)
        y = max(min(y, self.image_shape[1] - 1), 0)

        return np.array([y, x], dtype=PIXEL_COORDINATES_DTYPE)

    def __determine_scale_and_center(self, filename: str) -> \
                                     Tuple[np.ndarray, np.ndarray]:
        """
        Determine the degree-to-pixel ration for this image and the coordinates
        of its center.

        :param filename: the name of the fundus image
        :type filename: str
        :return: (y, x) scale and (y, x) center position
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        # check if we can determine scale from it: HRA - 30x30 degrees,
        # OCT - 3x3 or 6x6 mm
        if 'HRA' in filename:
            size_in_degrees = FUNDUS_IMAGE_SIZES_IN_DEGREES['HRA']
        elif 'OCTA' in filename:
            if 'HD' in filename:
                size_in_degrees = FUNDUS_IMAGE_SIZES_IN_DEGREES['OCTA_HD']
            else:
                size_in_degrees = FUNDUS_IMAGE_SIZES_IN_DEGREES['OCTA']
        else:
            size_in_degrees = None

        # if size in degrees can be determined based on the type of the image,
        # then do it
        if size_in_degrees is not None:
            size_in_degrees = np.array(size_in_degrees)
            self.image_shape_in_degrees = size_in_degrees
            return size_in_degrees / self.image_shape, self.image_shape // 2

        # otherwise ask the user to put two marks with known coordinates
        else:
            messagebox.showinfo(
                title='Cannot determine size of the image',
                message='This file is not HRA, OCTA HD or OCTA image (judging by '\
                        'its name). The scale of the image will '
                        'be determined manually.\n\n' \
                        'Please, click with left mouse button first on the center '\
                        'of the eye and then on the center of '
                        'the optic disc.\n\n'
                        'Please, note, then we consider the distance between '\
                        f'them to be {OPTIC_DISC_DISTANCE :.01f} '
                        'degrees. If the actual distance is different, then the '\
                        'grid will be placed incorrectly and '
                        'the shown zoomings might not be the actual representation '\
                        'of the AOSLO views.'
            )
            # self.__shutdown()
            size_in_degrees = FUNDUS_IMAGE_SIZES_IN_DEGREES['HRA']
            size_in_degrees = np.array(size_in_degrees)
            self.image_shape_in_degrees = size_in_degrees
            return size_in_degrees / self.image_shape, self.image_shape // 2

    def __mark_location(self, event: tk.Event) -> None:
        """
        On right mouse button click, a point will be marked as the very center
        of the eye and a red point will be drwawn on the fundus.

        :param event: the right click of the mouse
        :type event: tk.Event
        """

        # delete if the center of the image is already drawn and save the position
        # of the right-click
        if self.draw_center:
            self.image_widgets.delete("circle")
        self.center_loc = [event.x, event.y]

        # draw a 3x3 circle around the clicked position and get the position
        # in pixels
        self.draw_center = self.image_widgets.create_oval(event.x-3, \
            event.y-3, event.x+3, event.y+3, fill="red", tags="circle")
        mouse_coordinates_in_pixels = \
            self.__convert_mouse_coordinates_to_image_pixels(event)
        if np.any(mouse_coordinates_in_pixels is None):
            # the click is outside of the image widget
            return

        # get the location of the corresponding AOSLO image
        self.__get_view_location(mouse_coordinates_in_pixels)

    def __display_mouse_coordinates_in_degrees(self, coord: np.ndarray) -> None:
        """
        Display the mouse coordinates in degrees of the image on the right frame
        of the GUI window

        :param coord: coordinates of the mouse click
        :type coord: np.ndarray
        """

        # destroys the previous label if it exists and write the new one
        if self.label_center:
            self.label_center.destroy()
        self.label_center = tk.Label(self, bg='orange', text="Coordinates "\
            f"of the center in degrees: {np.flip(coord)}")
        self.label_center.pack(in_=self.frame_right, padx=10, pady=10)

    def __get_view_location(self, mouse_coordinates_in_pixels: np.ndarray):
        """
        Get the coordinates of the fundus image

        :param mouse_coordinates_in_pixels: (y, x) chosen coordinates in pixels
        :type mouse_coordinates_in_pixels: np.ndarray
        """

        # convert coordinates needed from pixels to degrees
        mouse_coordinates_in_degrees = (mouse_coordinates_in_pixels - \
            self.image_center) * self.degrees_per_pixel

        return self.__display_mouse_coordinates_in_degrees(mouse_coordinates_in_degrees)

    def __display_component_info(self, i: int) -> Tuple[tk.Label, tk.Label]:
        """
        Display the component informations on the right part of the GUI

        :param i: the index of the component that is displayed
        :type i: int
        :return: the label with the min and maxs and the location center of the
                 component
        :rtype: Tuple[tk.Label, tk.Label]
        """

        # find and display the component informations
        matched_chain = self.matched_chains[i]
        locations_present = [self.locations[j].tolist() for j in matched_chain]
        min_max = f"min_x = {np.min(np.array(locations_present)[:,0])}, "\
            f"min_y = {np.min(np.array(locations_present)[:,1])}, "\
            f"max_x = {np.max(np.array(locations_present)[:,0])}, "\
            f"max_y = {np.max(np.array(locations_present)[:,1])}"
        label1 = tk.Label(self, bg='orange', text=min_max, font=("Arial", 16))
        label1.pack(in_=self.frame_right, padx=20, pady=20)
        label2 = tk.Label(self, bg='orange', text="Location center: " \
            f"{[self.chain_centers[i][1], self.chain_centers[i][0]]}", \
            font=("Arial", 16))
        label2.pack(in_=self.frame_right, padx=20, pady=20)

        return label1, label2

    def __display_loading(self):
        """
        Display a loading text so that doctor wait a bit for the image to be
        shown
        """

        label = tk.Label(self, text="Loading", font=("Arial", 40))
        label.pack(in_=self.frame_right, padx=20, pady=20)
        self.after(2000, lambda: label.destroy())
        self.update()

    def __display_end_gui(self):
        """
        Display a finish text so that doctors know they are finished and it will
        close the GUI
        """

        label = tk.Label(self, bg='red', text="FINISH, will close program " \
            "automatically", font=("Arial", 30))
        label.pack(in_=self.frame_right, padx=20, pady=20)
        self.update()
        time.sleep(4)
        self.__quit_program(save=True)
