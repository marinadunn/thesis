# Thesis
Repo for Marina Dunn's 2023 M.Sc. Engineering: Data Science Thesis research development at UC Riverside. Title: "Classifying Galaxy Morphologies Using Bayesian Neural Networks to Support Future Astronomical Survey." This contains code for applying Deep Learning techniques to a simulated astronomical data in preparation for the Vera C. Rubin Observatory’s Legacy Survey of Space and Time (LSST). 

## Goal: 
Build, compile, and train Convolutional Neural Network (CNN), and eventually Bayesian Neural Network (BNN), on simulated data from LSST to classify whether a galaxy is a spiral, elliptical, or has undergone a merger. 

### Steps:
1. Load data; define training, test, and validation sets, and explore data
2. Build a deterministic CNN in Tensorflow & Keras
3. Compile the CNN
4. Train the CNN to perform a classification task
5. Evaluate the results and visualize
6. Visualize feature maps for best model
7. Perform Transfer Learning with additional data
8. Build a probabilistic CNN in Tensorflow & Keras
9. Compile the CNN
10. Train the CNN to perform a classification task
11. Evaluate the results and visualize
12. Build a BNN in Tensorflow & Keras
13. Compile the BNN
14. Train the BNN to perform a classification task
15. Evaluate the results and visualize

### Imports:
`numpy` and `pandas` to handle array functions

`matplotlib` and `seaborn` for plotting data

`keras` and `tensorflow` for building the CNN

`sklearn` for utility functions

## Data
The first half of this project focuses on using simulated data for LSST. The LSST mock catalog was created from the Illustris TNG100 simulation from snapshots year 1, with high noise, and year 10, with low noise. Noise and point-spread function was added using the GalSim package as part of the DeepAdversaries project. Both the large raw data files as well as resized smaller image files can be accessed on [Zenodo](https://zenodo.org/record/5514180#.Ymb3zi-B2L2) and are in .npy file format. 

Data Distribution:

There are 33,557 total images. There are 14312 : 8151 : 11094 images of spirals, ellipticals and mergers, respectively.

Training set: 23,487 images (10,017 spiral, 5,705 elliptical, 7,765 mergers)

Test set: 6,715 images (2,863 spiral 1,631 elliptical, 2,221 mergers)

Validation set: 3,355 images (1,432 spiral, 815 elliptical, 1,108 mergers)

Images are classified and one-hot encoded into 3 categorical labels: spiral ('0'), elliptical ('1'), or merger ('2’). The distribution is the same for both years 1 and 10.

The second half of the project will focus on comparing model performance using simulated and real observational data from the Hubble Space Telescope. Development for this will start in Fall 2022 and can be found in the file:

`HST.ipynb`

## LSST Training
For the LSST simulated catalog data, several baseline deterministic Convolutional Neural Network models were built, compiled, trained, and evaluated for both year 1 and year 10 image datasets. The best model was then chosen to develop a probabilistic CNN, then a Bayesian CNN, all developed using Keras and TensorFlow. This can be found in the file:

`LSST.ipynb`

The architecture for the first deterministic model is designed similarly to that used for the [DeepMerge project](https://github.com/AleksCipri/deepmerge-public), which uses 3 convolutional layers with dropout, 3 pooling layers, flattening, and then 3 dense layers with the first 2 having weight regularization. This achieved a classification accuracy of 76% for low-noise images and 79% for noisy images, but instead uses Categorical Cross-Entropy loss and the Softmax activation function to make predictions for the 3 classes. 

Several methods were also used to try to prevent overfitting, including Batch Normalization, Dropout, L2 regularization for kernel regularizers, and the Early Stopping (monitoring validation loss), ModelCheckpoint, and ReduceLCOnPlateau callback functions.

## HST Training
TBD

## Requirements
This code was developed using Python 3., TensorFlow 2.9.0 (installed via [Apple Mac M1 chip instructions](https://developer.apple.com/metal/tensorflow-plugin/)), and TensorFlow Probability 0.16.0.
