# FRC-Net

This repository contains the code and data for the manuscript

### Fourier Ring Correlation and anisotropic kernel density estimation improve deep learning based SMLM reconstruction of microtubules

*by Andreas Berberich, Andreas Kurz, Sebastian Reinhard, Torsten Johann Paul, Paul Ray Burd, Markus Sauer, Philip Kollmannsberger*

doi xxx

This work was part of the M.Sc. thesis project of Andreas Berberich in the computational image analysis group of Philip Kollmannsberger at the Center for Computational and Theoretical Biology of the University of Würzburg, carried out between October 2018 and March 2020.

- The notebooks `Fig-XXX.ipynb` produce the Figures in the manuscript.
- `FRCNet-train.ipynb` and `FRCNet-predict.ipynb` can be used for training and inference of the neural network described in the manuscript.
- `frc_loss.py` and `aniso_kde.py` contain the implementations of FRC loss in Tensorflow and of the anisotropic KDE pre-filtering described in the manuscript.
- The `/data` directory contains training/validation data and the data to generate the figures.

All code is written in Python3 and requires numpy, scipy, matplotlib and tensorflow 2.0. If you find this code useful and want to use it in your own project, please cite our paper.

