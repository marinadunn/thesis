# Thesis
Repo for Marina Dunn's 2023 M.Sc. Engineering: Data Science Thesis research development at UC Riverside. Title: "Classifying Galaxy Morphologies Using Bayesian Neural Networks to Support Future Astronomical Survey."

This contains code for applying probabalistic deep learning techniques to a simulated catalog for the Vera C. Rubin Observatory’s Legacy Survey of Space and Time (LSST). The goal is to accurately classify these images into 3 galaxy morphology categories: spiral, elliptical and mergers.

## Data
The LSST mock catalog was created from the Illustris TNG100 simulation from snapshots year 1, with high noise, and year 10, with low noise. Noise and point-spread function was added using the GalSim package as part of the DeepAdversaries project. Both the large raw data files as well as resized smaller image files can be accessed on [Zenodo](https://zenodo.org/record/5514180#.Ymb3zi-B2L2) and are in .npy file format. 

The training set consists of 23487 images, the test set consists of 6715 images, and the validation set consists of 3355 images. For image labels, the image is classified into 3 categories: spiral ('0'), elliptical ('1'), or merger ('2’).

## Part 1
For the LSST simulated catalog data, several original baseline deterministic Convolutional Neural Network models were developed and tested, then a subsequent probabilistic CNN, then a Bayesian CNN, all developed using TensorFlow as part of the UC Riverside CS 235: Data Mining course (Spring 2022). These can be found in the file:

`CNN.ipynb`

Standard deterministic Convolutional Neural Network models were developed first to use as baseline models. The architecture for the first deterministic model is designed similarly to that used for the [DeepMerge project](https://github.com/AleksCipri/deepmerge-public), which achieved a classification accuracy of 76% for low-noise images and 79% for noisy images, but instead uses Categorical Cross-Entropy loss and the Softmax activation function to make predictions for the 3 classes.

Several methods were used to try to prevent overfitting, including Batch Normalization, Dropout, L2 regularization for kernel regularizers, and the Early Stopping, ModelCheckpoint, and ReduceLCOnPlateau callback functions.

## Requirements
This code was developed using TensorFlow 2.8.0 (installed via [Apple Mac M1 chip instructions](https://developer.apple.com/metal/tensorflow-plugin/)), and TensorFlow Probability 0.16.0.
