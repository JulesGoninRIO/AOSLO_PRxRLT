# """From https://stackoverflow.com/questions/44887576/how-can-i-create-a-drag-and-drop-interface"""
""" and https://python-forum.io/thread-2729.html """

import tkinter as tk
from tkinter import _tkinter
import cv2
import os
from PIL import Image, ImageTk
import PIL
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# global variable because we want all the components to end up with the same size
scale = 1.0

class DragManager():
    """
    Class to handle the connected components Image objects where the images can
    be drag and drop where we need to place them. We can also rotate and resize
    the first loaded component, change transparancy of components when shown but
    also when dragged.
    """

    def __init__(self, montage_dir: Path, canvas: tk.Canvas, image_name: str, \
                 image_height: int, image_width: int, xpos: int = None, \
                 ypos: int = None, transparancy: float = 0.7, \
                 dragged_transparancy: float = 0.75, zoom_allowed: bool = False, \
                 angle: int = 0):
        """
        Initialize the DragManager object with all the needed and optional
        parameters as follow. The zoom should be allowed only on the first component

        :param montage_dir: path where the components images are
        :type montage_dir: Path
        :param canvas: Tkinter canvas object to place the connected component on
        :type canvas: tk.Canvas
        :param image_name: the anme of the component to place
        :type image_name: str
        :param image_height: the height of the component (bigger if zoomed)
        :type image_height: int
        :param image_width: the width of the component (bigger if zoomed)
        :type image_width: int
        :param xpos: the center x position where to place the image, defaults to None
        :type xpos: int, optional
        :param ypos: the center y position where to place the image, defaults to None
        :type ypos: int, optional
        :param transparancy: the transparancy scale when the image is placed,
                             defaults to 0.7
        :type transparancy: float, optional
        :param dragged_transparancy: the transparancy scale when the image is moved,
                                     defaults to 0.75
        :type dragged_transparancy: float, optional
        :param zoom_allowed: whether we have the zoomed view or not, defaults to False
        :type zoom_allowed: bool, optional
        :param angle: the angle of the component, defaults to 0
        :type angle: int, optional
        """

        # initialize the objects from the parameters given
        global scale
        self.canvas = canvas
        self.image_name = image_name
        self.montage_dir = montage_dir

        # xpos and ypos are the center of the images, not the top left
        if xpos:
            self.xpos = xpos
        else:
            self.xpos = image_height/2
        if ypos:
            self.ypos = ypos
        else:
            self.ypos = image_width/2

        self.image = None
        self.zoom_allowed = zoom_allowed
        self.transparancy = transparancy
        self.dragged_transparancy = dragged_transparancy
        self.angle = angle

        # read and resize the component so that we don't shear it
        # we need to have the alpha channel present so that we see the fundus
        # image below the component one
        im = cv2.imread(os.path.join(self.montage_dir, image_name))
        ih_adapted = (im.shape[0]*image_width)/im.shape[1]
        alpha = np.sum(im, axis=-1) > 0
        alpha = np.uint8(alpha * round(self.transparancy*255))
        res = np.dstack((im, alpha))
        self.base_image = Image.fromarray(res).resize((round(image_width), \
            round(ih_adapted)))
        self.image = Image.fromarray(res)
        if self.angle != 0:
            self.image = self.image.rotate(self.angle, expand=1)
            ih_adapted = (self.image.size[1]*image_width)/self.image.size[0]
        self.size = round(image_width * scale), round(ih_adapted * scale)
        self.image = self.image.resize((self.size))

        # copy the image once it has the good size and angle so that ... and display it
        self.non_transparant = self.image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_obj= canvas.create_image(
            self.xpos, self.ypos, image=self.tk_image)

        def __leftKey(event: tk.Event):
            """
            Bind the left key arrow of the keyboard to move the image by one
            pixel to the left

            :param event: when the left arrow key is pressed
            :type event: tk.Event
            """

            # move and recreate the canvas object
            self.__move_canvas(dx = -1)

        def __rightKey(event: tk.Event):
            """
            Bind the right key arrow of the keyboard to move the image by one
            pixel to the right

            :param event: when the right arrow key is pressed
            :type event: tk.Event
            """

            # move and recreate the canvas object
            self.__move_canvas(dx = 1)

        def __topKey(event: tk.Event):
            """
            Bind the top key arrow of the keyboard to move the image by one
            pixel to the top

            :param event: when the top arrow key is pressed
            :type event: tk.Event
            """

            # move and recreate the canvas object
            self.__move_canvas(dy = -1)

        def __bottomKey(event: tk.Event):
            """
            Bind the bottom key arrow of the keyboard to move the image by one
            pixel to the bottom

            :param event: when the bottom arrow key is pressed
            :type event: tk.Event
            """

            # move and recreate the canvas object
            self.__move_canvas(dy = 1)

        def __rotate_left(event: tk.Event):
            """
            Bind the left control (ctrl) arrow of the keyboard to rotate the image by one
            degree anti clock-wise

            :param event: when the left ctrl key is pressed
            :type event: tk.Event
            """

            # rotate and recreate the canvas object
            self.__rotate_image(1)

        def __rotate_right(event: tk.Event):
            """
            Bind the right control (ctrl) arrow of the keyboard to rotate the image by one
            degree clock-wise

            :param event: when the right ctrl key is pressed
            :type event: tk.Event
            """

            # rotate and recreate the canvas object
            self.__rotate_image(-1)

        # bind tags build beforehand
        self.__bind_drag_drop_tags()
        self.canvas.bind('<Left>', __leftKey)
        self.canvas.bind('<Right>', __rightKey)
        self.canvas.bind('<Up>', __topKey)
        self.canvas.bind('<Down>', __bottomKey)
        self.canvas.bind('<Control_L>', __rotate_left)
        self.canvas.bind('<Control_R>', __rotate_right)
        self.canvas.focus_set()

        #windows scroll
        if self.zoom_allowed:
            self.canvas.bind("<MouseWheel>", self.__zoomer)

        # tag that say when the image is dragged so that it render it more
        # transparent and show the moved image
        self.move_flag = False

    def __move(self, event: tk.Event):
        """
        Functions that allows the movement of the image when it is dragged

        :param event: left-click of the mouse to drag the image
        :type event: tk.Event
        """

        # if the drag has already started, the move_flag will be set to True and
        # we move the image following the mouse position
        if self.move_flag:
            new_xpos, new_ypos = event.x, event.y
            self.canvas.move(self.image_obj,
                new_xpos-self.mouse_xpos , new_ypos-self.mouse_ypos)
            self.mouse_xpos = new_xpos
            self.mouse_ypos = new_ypos

        # here it is the first click of the mouse when we drag the image so w
        # need to reload a more transparant image so that we better see the
        # fundus image underlying
        else:
            self.move_flag = True
            self.image = Image.fromarray(np.dstack((np.array(self.image)[:,:,0:3], \
                np.uint8((self.dragged_transparancy*np.array(self.image)[:,:,-1]).round()))))
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.image_obj= self.canvas.create_image(
                self.xpos, self.ypos, image=self.tk_image)
            self.canvas.tag_raise(self.image_obj)
            self.mouse_xpos = event.x
            self.mouse_ypos = event.y

    def __release(self, event: tk.Event):
        """
        Functions called when we release the left-click of the mouse so that
        we reload the less transparant image and place it where the image was
        released. If you want to do minor adjustements afterwards use the arrow
        keys.

        :param event: left-click release of the mouse
        :type event: tk.Event
        """

        # reload and display the image released
        self.image = self.non_transparant.copy()
        self.image = self.image.rotate(self.angle, expand=1)
        self.tk_image = ImageTk.PhotoImage(self.image)
        coords = self.canvas.coords(self.image_obj)
        self.image_obj= self.canvas.create_image(
            coords[0], coords[1], image=self.tk_image)
        self.__bind_drag_drop_tags()
        self.move_flag = False
        self.xpos = coords[0]
        self.ypos = coords[1]

    def __zoomer(self, event: tk.Event):
        """
        Allows the mouse roll to increase (roll to the top of the mouse) or
        decrease (roll bottom) the image size.
        Warning: it can be done only for the first component loaded

        :param event: mouse roll of the mouse
        :type event: tk.Event
        """

        # change the global scale so that all the following component will
        # be the same size
        global scale

        # scroll top to zoom in, bottom to zoom out
        if (event.delta > 0):
            scale *= 1.01
        elif (event.delta < 0):
            scale *= 0.99
        self.__redraw()

    def __redraw(self):
        """
        Redraw the scaled image object
        """

        global scale
        coords = self.canvas.coords(self.image_obj)

        # keep image coordinates of the center and only rescale
        self.xpos = coords[0]
        self.ypos = coords[1]
        if self.image_obj:
            self.canvas.delete(self.image_obj)
        iw, ih = self.size[0], self.size[1]
        new_size = round(iw * scale), round(ih * scale)
        self.image = self.base_image.copy()
        self.image = self.base_image.resize(new_size)
        self.non_transparant = self.image.copy()
        self.image = self.image.rotate(self.angle, expand=1)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_obj = self.canvas.create_image(coords[0], coords[1], image=self.tk_image)
        self.canvas.tag_raise(self.image_obj)
        self.__bind_drag_drop_tags()

    def get_coordinates(self) -> Tuple[float, float, Tuple[int, int], float, int]:
        """
        Return the component informations when going to the next compoenent so
        that the scale is respected and we can write out the results of the
        component location

        :return: the component informations
        :rtype: Tuple[float, float, Tuple[int, int], float, int]
        """

        # get and returns the informations
        # if the zoom is allowed, it means it is the first component and we now
        # need to disallow the zooming
        global scale
        if self.zoom_allowed:
            self.canvas.unbind("<MouseWheel>")
        coords = self.canvas.coords(self.image_obj)
        self.xpos = coords[0]
        self.ypos = coords[1]

        return self.xpos, self.ypos, self.image.size, scale, self.angle

    def delete(self):
        """
        Delete the binding of the zoom and the moving binding so that we can now
        place the next component
        """

        # oinly the first component can have the zoom option
        if self.zoom_allowed:
            self.canvas.unbind("<MouseWheel>")
        if self.image_obj:
            self.canvas.delete(self.image_obj)

    def __move_canvas(self, dx: int = 0, dy: int = 0):
        """
        Move a and recreate a canvas object

        :param dx: the difference in x to displace, defaults to 0
        :type dx: int, optional
        :param dy: the difference in y to displace, defaults to 0
        :type dy: int, optional
        """

        # move and rebuild image object
        self.canvas.move(self.image_obj, dx, dy)
        self.__rebuild_canvas()

    def __rotate_image(self, angle: int = 0):
        """
        Rotate an image by the angle given

        :param angle: the angle to rotate, defaults to 0
        :type angle: int, optional
        """

        # rotate and rebuild image object
        self.angle += angle
        self.image = self.image.rotate(angle, expand=1)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.__rebuild_canvas()

    def __rebuild_canvas(self):
        """
        Rebuild the Canvas object that has been changed
        """

        # Build and bind buttons needed to drag and drop the image
        coords = self.canvas.coords(self.image_obj)
        self.canvas.delete(self.image_obj)
        self.image_obj= self.canvas.create_image(
            coords[0], coords[1], image=self.tk_image)
        self.__bind_drag_drop_tags()

    def __bind_drag_drop_tags(self):
        """
        Bind tags of drag and drop to the Canvas object
        """
        self.canvas.tag_bind(self.image_obj, '<Button1-Motion>', self.__move)
        self.canvas.tag_bind(self.image_obj, '<ButtonRelease-1>', self.__release)