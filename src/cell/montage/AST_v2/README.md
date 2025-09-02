# GUI AST_V2 (= Montaging GUI)

## How to Run it

Open the folder:

```
cd aoslo_pipeline/src/PostProc_Pipe/Montaging/AST_v2
```

And run the main script:

```
python main.py
```

or run it via the InputGUI. Open the folder:

```
cd aoslo_pipeline/src/InputGUI
```

Run the main script:

```
python main.py
```

And when the GUI is open, click on the **Montaging GUI** button.

## How to use it

This GUI is a prototype to help correcting montages from Matlab by allowing the user to place the unconnected components relative to locations in the retina from the fundus image. You should be able to place the components by focussing on placing correctly the vessels of the AOSLO image compare to the fundus vessels. Here are the instructions for using this GUI:

1. Run the GUI by launching the .exe file

2. A Windows File Explorer will be open and you should first give the location of the fundus image file by double clicking on it. It will load the mirorred (left/right inversed to match AOSLO reflective way of capturing images) fundus image.

Fundus selection example:
![Alt text](../../../readme_images/montaging_gui/fundus.png?raw=true "Fundus selection example")

3. The same is to do for the total montaged image (**all_ref_combined_m1.tif**) that is the output from Matlab. **WARNING**: the total montaged image should be in the same folder as the multiple unconnected components so that they are being loaded when needed. Also please use the *Confocal* images to correct the montage, it will be easier for you.

Montaged component selection example:
![Alt text](../../../readme_images/montaging_gui/montaged.png?raw=true "Montaged component selection example")

4. The GUI will display the fundus image and you will be able to **RIGHT-CLICK** to select the very center of the retina. Once the click is done, a red circle will be drawn at the location of the very center of the retina. Feel free to right-click again if it is not precise enough. Once done, click on the button to validate this location.

Center selection example:
![Alt text](../../../readme_images/montaging_gui/center.png?raw=true "Center selection example")

5. Once the center location validated, the first component will appear after some loading time (approximately 2s to wait each time a component is loaded). Place this component at its location by using the following tools:
	- **Drag and drop**: Left-click on the image and maintain the click to deplace the image and release the click to release the image. When you're moving the image around, the image will be more transparant.
	- Use **keyboard arrows (up, down, left, right)** to displace the components more precisely.
	- **Zoom and dezoom**: Use the mouse roll or two-fingers on your pad to enlarge or reduce the connected component size. WARNING: you will only be able to adjust size of the first component: the others will be loaded accordingly to the first component's size for consistency.
	- **Right and left Ctrl buttons**: Use the ctrl buttons to rotate each components in specific direction in order to better match the vessels of the funuds. You can rotate every component independently. The rotation induce a bit of blurring in the image but drag and drop it and the blur will disappear.
	- **Transparancy** levels: You can choose to have the connected component more or less transparant by adjusting the transprancy buttons (+ or -) if the fundus image is darker or brighter. The step size is 0.1 and starts at 0.7.
	- Image **dragged transparancy** levels: You can also choose to add or remove the same way the transparancy when you drag the image. The step size is 0.25 and starts at 0.75.
	- Component's information: In red there will be the relative location of the compoenent loaded. The location center is in coordinates [x,y], where x is the horizontal axis going from left to right and y is the vertical axis, going from top to bottom. WARNING: it is possible that the component loaded is at the totally opposite of where it should be placed: this is a bug from the MATLAB code, you should pay attention to the location center of each component because the relative location loaded could be wrong.

Component placement example:
![Alt text](../../../readme_images/Montaging2.png?raw=true "Component placement example")

6. Once the component is well-placed, press the **validate** image button, or the **suppress** compoenent button, whether you want to keep this connected compoenent or you want to suppress it (e.g if you cannot place the component or don't want the component in the result). Pressing one of those 2 buttons will load the next unconnected component after the loading time. Also you can choose to **Do Later** the component if you don't know where to place the component so that you can do it at the end when all other components are placed.

7. Place all the others component and validate or suppress each. **WARNING**: If you notice you have wrongly place a component, you can press the **cancel** button. This will suppress the previsouly saved (or suppressed) component and you will be able to re-do it when all others connected compoenents have been placed. You can use this button multiple time if you notice a misplaced component placed a while ago. When you re-do the cancelled components at the end, please do not make a mistake because you won't be able to press cancel again, you will have to start all over again.

8. Once finished, a statement will say that you have finished and will close AUTOMATICALLY the GUI. **PLEASE DON'T CLOSE IT BY CLICKING ON THE UPPER-RIGHT CROSS BECAUSE YOUR WORK WON'T BE SAVED.**

9. Letting close the GUI by itself will save the positions of the components in a .csv file called locations.csv (if not changed), in the same folder as the connected components. This file will be use in the pipeline to have locations of the images in the retina.

Output csv example:
![Alt text](../../../readme_images/montaging_gui/locations.png?raw=true "Output csv example")

Thank you!

Further improvement for the next developper:
- Add explanations on the GUI at each step of the GUI so that you don't need to read this file before using the GUI.
- Make it less ugly and more intuitive.
- Load images at ~correct location -> take code from Montager.py in the AOSLO Pipeline and look to correct the montaging before loading the components so that you know what should be there relative locations.