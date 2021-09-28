# FRCnet

This repository contains the code and data for

### Fourier Ring Correlation and anisotropic kernel density estimation improve deep learning based SMLM reconstruction of microtubules

*by Andreas Berberich, Andreas Kurz, Sebastian Reinhard, Torsten Johann Paul, Paul Ray Burd, Markus Sauer, Philip Kollmannsberger*

This work was part of the MSc thesis project of Andreas Berberich in the Computational Image Analysis group of Philip Kollmannsberger at the [Center for Computational and Theoretical Biology](https://www.biozentrum.uni-wuerzburg.de/cctb/cctb/) of the University of Würzburg, carried out between October 2018 and March 2020.

This repository contains the following files:

- The notebooks `Fig{1,2,3,4}.ipynb` reproduce the figures in the manuscript.
- `FRCNet.ipynb` contains the code to train the neural network.
- `frc_loss.py` and `aniso_kde.py` contain the implementation of the tensorflow FRC loss and the anisotropic KDE filtering.
- The `/data` directory contains training/validation data and the data to generate the figures.

All code was written in Python3 using numpy, scipy, scikit-image, pandas, matplotlib and tensorflow 2.4.1. If you find this code useful and want to use it in your own project, please cite our paper.

To run the notebooks, create a new conda environment and install the dependencies:

`conda install -c anaconda tensorflow-gpu=2.4.1 jupyter matplotlib pandas scikit-image`

and then install `tensorflow-addons` (required for on-GPU image rotation) using `pip`:

`pip install tensorflow-addons`

The raw localization data and trained models can be downloaded from [here](https://drive.google.com/file/d/1SjAWtNEiLYu2My7Z2znpoEWxt3kskAea/view?usp=sharing) and should be placed in the `/data` folder.


