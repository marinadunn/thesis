# Thesis
Repo for Marina Dunn's 2023 M.Sc. Engineering: Data Science Thesis research development at UC Riverside. Title: "Classifying Galaxy Morphologies Using Bayesian Neural Networks to Support Future Astronomical Survey." This contains code for applying Deep Learning techniques to a simulated astronomical data in preparation for the Vera C. Rubin Observatory’s Legacy Survey of Space and Time (LSST). 

## Goal: 
Build, compile, and train Convolutional Neural Network (CNN), and eventually Bayesian Neural Network (BNN), on simulated data from observing snapshots year 1 and 10 for LSST, and perform multi-class classification task, determining whether image is a spiral galaxy, elliptical galaxy, or galaxy merger. 

## Steps:
1. Load numpy data files
2. Scale data
3. Cast data type
4. Exploratory Data Analysis/Critical diagnostics
5. Define deterministic CNN model architecture
6. Compile model
7. Train model
8. Evaluate model on test set & smaller sub-test set
9. Plot training history (training and validation accuracy and loss)
10. Plot confusion matrix & print classification report
11. Save model data
12. Fine-tune & repeat until best model found **(Current Step)**
13. Visualize feature maps for best deterministic model
14. Perform Transfer Learning with additional data
15. Define probabilistic CNN model architecture
16. Compile model
17. Train model
18. Evaluate model on test set & smaller sub-test set
19. Plot training history (training and validation accuracy and loss)
20. Plot confusion matrix & print classification report
21. Save model data
22. Define Bayesian NN model architecture
23. Compile model
24. Train model
25. Evaluate model on test set & smaller sub-test set
26. Plot training history (training and validation accuracy and loss)
27. Plot confusion matrix & print classification report

## Data
This project focuses on using simulated data for LSST, created from the Illustris TNG100 simulation, with observing snapshots from year 1 (high noise) and year 10 (low noise). Noise and point-spread function was added using the GalSim package as part of the DeepAdversaries project. Image labels are one-hot encoded into 3 categories: Spiral ('0'), Elliptical ('1'), or Merger ('2’). The distribution is the same for both years 1 and 10. Both the large raw data files as well as resized smaller image files can be accessed on [Zenodo](https://zenodo.org/record/5514180#.Ymb3zi-B2L2) in .npy file format. 

* Training set: 23,487 images (10,017 Spirals; 5,705 Ellipticals; 7,765 Mergers)
* Testing set: 6,715 images (2,863 Spirals; 1,631 Ellipticals; 2,221 Mergers)
* Validation set: 3,355 images (1,432 Spirals; 815 Ellipticals; 1,108 Mergers)
* Total images: 33,557 (Splitting fraction 70:20:10)

Class Balance: 
* Training - Spirals: 0.7816, Ellipticals: 1.3723, Mergers: 1.0082
* Test - Spirals: 0.7818, Ellipticals: 1.3724, Mergers: 1.0078
* Validation - Spirals: 0.78096, Ellipticals: 1.3722, Mergers: 1.0093

## Training
To begin, several baseline deterministic Convolutional Neural Network models were built, compiled, trained, and evaluated for both year 1 and year 10 image datasets. The best model was then chosen to develop a Bayesian CNN. Model architecture was built using Keras and TensorFlow.

Model development can be found in the Jupyter notebook `LSST.ipynb`

# Deterministic CNN Model Architecture:
* Loss functions used:
  * Categorical cross-entropy (because this is a multi-class classification problem, and the labels are one-hot encoded)
* Activation functions used:
  * ReLU (All but last layer)
  * Softmax (Last Dense layer)
* Optimizers used:
  * Adam
  * Adam weighted
* Regularization techniques used (to try to prevent overfitting):
  * Batch Normalization
  * Dropout
  * L2 kernel regularization
* Callback Functions used:
  * Early Stopping (monitoring validation loss)
  * ModelCheckpoint (save best weights only)

## Requirements
This code was developed using Python 3.10, TensorFlow 2.9.0 (installed via [Apple Mac M1 chip instructions](https://developer.apple.com/metal/tensorflow-plugin/)), and TensorFlow Probability 0.16.0.
