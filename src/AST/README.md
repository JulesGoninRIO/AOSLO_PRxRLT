# Aquisition Support Tool

This Aquisition Support Tool (AST) has been created to help the optometrist or
doctor while capturing AOSLO images.

Developped by Mikhail Tsarytsin

## How to Run it

Open the folder:

```bash
aoslo_pipeline/AST
```

And run the main script:

```bash
python main.py
```

or run it via the InputGUI. Open the folder:

```bash
aoslo_pipeline/input_GUI
```

Run the main script:

```bash
python main.py
```

And when the GUI is open, click on the AST GUI button.

## How to use it

The GUI will ask you to open a subject's  fundus image and will display it on the
main window with a grid that allows to left-click on it to zoom on the part of th
fundus image. To select grid cells, right-click on them. If you want to remove the
grid cell previsouly selected, right-click on it again.

![Alt text](../readme_images/AST.png?raw=true "AST use case example")

(real time GUI) determine the position in degrees of a location on the fundus
(useful for vessels, photoreceptors have fixed pattern)
    - import fundus to be used as a localizer
      - operator opens his/her laptop
      - opens topcon/heyex online reader
      - takes screenshot
    - enter: right or left eye
    - click on Macula, click on OD. Calculate scaling factor from this
      - or, in case of OCTA, provide hard choice: 3x3 or 6x6
    - put a grid on the localizer
    - if operator clicks on a square, GUI shows
      - AOSLO coordinates
      - preview of what the AOSLO capture is expected to look like
      - save preview
    - square turns green
    (image is taken on AOSLO, process restarts)
