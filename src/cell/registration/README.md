# Registring step

This step allows to find the overlap between every pair of Confocal and OA850nm images.

## Parameters explanation

1. **Do Registring Analysis**: Whether or not to run the total Registring analysis that look for the overlap metrics by the traditional method POC but also with Enhanced Cross-Correlation (ECC) and Mutual Information (MI) and on different method (Confocal (traditional), CalculatedSplit and DarkField).

2. **Registring Directory Name**: Name of the directory where the results of the Registring algorithm (and analysis if selected) will be found.

## Run the registring step

1. Run via the **InputGUI**: Go to the specific directory via: `cd aoslo_pipeline/src/InputGUI` And run: `python main.py`. Then make sure the tickbox **Do Registring** is checked, that you have specified the parameters you want in the **REGISTRING** subsection and that you have selected the correct input folder. Then just press the **Run pipeline!** and follow the instructions.

2. Run via the *config.txt*:Write in the config file that can be found in *aoslo_pipeline/src/PostProc_Pipe/Configs/config.txt* the specific parameters that you want to run, especially everything under the [Registring] section (*__do_montaging* must be set to **True** to run this step of the pipeline). Then go to the specific folder: `cd aoslo_pipeline/src/PostProc_Pipe` and run `python Start_PostProc_Pipe.py`.

## Substeps of Registring

1. The images are scanned to look for Confocal-OA850nm pairs that will be run through the registration algorithm.

2. The image pairs offset are computed with help of the POC (Phase-Only Correlation) algorithm and plots are created with the results so that we can visually inspect the pairs where the offset is too much different from others.

## Considerations about this step

The Registrationn analysis was mainly created to look whether using other modality images or other algorithm as POC would give better results than Confocal with POC. But POC works better so if you want to find the offset just run the code without the analysis.

## Deeper Explanation of the Maths behind the Registring step

We aim to compare cone pixel intensities in Confocal and OA850nm: since those two modalities are captured with a spatial offset, I registered them. As the OA850nm modality has just been developed, there is no off-the-shelf algorithm able to find automatically the offset between those modalities. I tried different techniques such as keypoints detector like SIFT followed by keypoints matching algorithm such as RANSAC. I also tried SuperPoint combined with SuperGlue [1], also followed by RANSAC. The shift between the modalities is set to around 10 pixels, so I could have reduced the search to a neighborhood around the expected location. Furthermore, there can only be translation between modalities, so I could have forced algorithms to find affine transformwith only translation. Nevertheless, as the algorithms do not detect those features automatically, I did not force them to do so and preferred to find an algorithm able to find those affine transformations. The difficulty resides in the fact that even if both modalities are captured by the Confocal pinhole, they don’t have the same light intensities and thus the resulting pixel intensity at an exact location is not the same between images.

I then tried the ECC [2] algorithm based on ZNCC and a POC algorithm [3] based on the correlation in the Fourier phase domain. I also analyzed different registration metrics on the overlapping regions, notably the NCC, NMI, NMSE and SSIM, in the attempt to isolate outliers where the registration had failed. Those outliers were found using KMeans with 2 clusters, one capturing the registrations that worked, the other cluster capturing the outliers. After this analysis, I opted for the POC algorithm. To handle outliers that were wrongly registered, I also ran POC between CalculatedSplit and OA850nm, as well as between DarkField and OA850nm. Finally, I used the overlapping metrics cited above to correct the remaining outliers.

Phase-Only Correlation (POC) [3] is a sub-pixel image matching technique. It uses the closed-formanalytical model to fit the two-dimensional pixels’ numerical data [4]. The algorithmalso uses a Peak Evaluation Formula (PEF) that directly estimates the correlation peak location from the POC function, to reduce computation time without sacrificing image matching accuracy. The POC method works the following:

1. An analytical function fitting technique to estimate the location of the correlation peak.
2. A windowing technique to eliminate the effect of periodicity in 2D Discrete Fourier Transform (DFT).
3. A spectrum weighting technique to reduce the effect of aliasing and noise.

[1] Paul Edouard Sarlin, Daniel Detone, Tomasz Malisiewicz, and Andrew Rabinovich. “SuperGlue: Learning FeatureMatching with Graph Neural Networks”. In: Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (2020), pp. 4937–4946. I S SN: 10636919. DOI: 10.1109/ CVPR42600.2020.00499.

[2] Georgios D Evangelidis and Emmanouil Z Psarakis. “Parametric image alignment using enhanced correlation coefficient maximization”. In: IEEE transactions on pattern analysis and machine intelligence 30.10 (2008), pp. 1858–1865.

[3] Sei Nagashima, Takafumi Aoki, Tatsuo Higuchi, and Koji Kobayashi. “A subpixel image matching technique using phase-only correlation”. In: 2006 International Symposium on Intelligent Signal Processing and Communications. IEEE. 2006, pp. 701–704.

[4] Kenji Takita, Takafumi Aoki, Yoshifumi Sasaki, Tatsuo Higuchi, and Koji Kobayashi. “High-accuracy subpixel image registration based on phase-only correlation”. In: IEICE transactions on fundamentals of electronics, communications and computer sciences 86.8 (2003), pp. 1925–1934.

## For developers:

About Registring: SIFT, Super-Glue or other techniques that use keypoints detectors do not work (most have been tested) because they cones look way too much like each other. That is why we use POC which looks more to the structure and is able to find the offset between the images.
