# LSST Galaxy Morphology Classification Using Bayesian Neural Networks
This repository contains development code and documentation for my M.S. Engineering: Data Science thesis research (final project expected June 2023) at UC Riverside, entitled "Galaxy Morphology Classification Using Bayesian Neural Networks for the Legacy Survey of Space and Time (LSST)." UCR Project Advisor: Dr. Bahram Mobasher, Deep Skies Lab Research Advisors: Dr. Aleksandra Ciprijanovic (Fermilab, U Chicago), and Dr. Brian Nord (Fermilab, U Chicago, Kavli Institute for Cosmological Physics, MIT)

## Overview:
In the coming decade, new observatories will begin operations, enabling extensive large-scale, multi-wavelength sky surveys for significant astrophysical inquiries. A central focus is understanding galaxy formation and evolution, exemplified by the upcoming Legacy Survey of Space and Time (LSST), with observations starting in 2024. LSST will yield vast data volumes, with many images and catalogs being produced on short timescales, demanding innovative solutions for efficient processing. Machine Learning techniques, such as convolutional neural networks, offer the ability to improve and automate many of these data processes. To prepare for LSST-like surveys, we utilize deep neural networks in order to classify 3 galaxy morphologies in simulated LSST-like images of different quality, accounting for realistic observing degradations like noise. In addition to standard convolutional networks, we explore utilizing Bayesian neural networks, capable of quantifying uncertainty in the parameters of a model and, consequently, of its predictions. Notably, we find that networks trained on noisy early-release data struggle with less noisy subsequent-release data, but transfer learning techniques mitigate this. This emphasizes the need for realistic simulated data in machine learning model development, bridging the gap between simulated and real observations. Further research is warranted, especially refining Bayesian models to provide accurate uncertainties for simulated LSST-like images. As such methods become standard for future surveys, robust models must transition smoothly between simulated and real data, accommodating successive data releases from the same telescopes.

## About:
We first develop a standard deterministic Convolutional Neural Network (CNN) model, and eventually a fully probabilistic Bayesian Neural Network (BNN) model trained on a simulated mock data catalog of galaxies created from the Illustris TNG100 simulation as part of the [DeepAdversaries project](https://github.com/AleksCipri/DeepAdversaries) representing observing galaxies at various years with the upcoming Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST), where observations at Year 1 are noisier, and Year 10 are less-noisy. We attempt to accurately classify the morphologies of these galaxies as either 'Spiral', 'Elliptical', or 'Merger.'

## Data
The simulated mock imaging LSST data catalog used was created from the [Illustris TNG100 simulation](https://www.illustris-project.org) as part of the [DeepAdversaries project](https://github.com/AleksCipri/DeepAdversaries), with observational noise and PSF added using the [GalSim package](https://github.com/GalSim-developers/GalSim). Both the original raw data files, as well as smaller subsets, can be accessed on the [DeepAdversaries Zenodo page](https://zenodo.org/record/5514180#.Ymb3zi-B2L2) in .npy file format. The mock image labels are one-hot encoded into 3 classes: Spiral ('0'), Elliptical ('1'), or Merger ('2’). Data augmentation such as rotation and horizontal/vertical flipping has already been performed for the galaxy merger class to improve class imbalance, and augmented images are already saved in data files.

Data Distribution:
* Training set: 23,487 images (10,017 Spirals, 5,705 Ellipticals, 7,765 Mergers)
* Testing set: 6,715 images (2,863 Spirals; 1,631 Ellipticals; 2,221 Mergers)
* Validation set: 3,355 images (1,432 Spirals; 815 Ellipticals; 1,108 Mergers)
* Total: 33,557 images (Splitting fraction 70% train : 20% test : 10% validation)

Example Images:
![Example Galaxy Images for Year 1 and Year 10](https://github.com/marinadunn/thesis/blob/main/Plots/EDA/example_images.jpg "Example Images of Year 1 & Year 10 Galaxies")

## Installation
This code was developed using Python 3.10, Tensorflow-Probability 0.16.0, and Tensorflow 2.9.0. It is recommended to create a new Python virtual environment before running this code. All dependencies can be found in `requirements.yml`.

To clone this repository and set up the environment, run the following commands in a terminal:
```
git clone https://github.com/marinadunn/thesis.git
cd thesis
python3 -m venv [envname]
source [envname]/bin/activate
python3 -m pip install -r requirements.txt
```

## Usage
Current branch: `main`

Model development notebooks are separated by standard convlolutional neural networks (CNNs) and Bayesian neural networks (BNNs), and whether classification models are trained on noisier Year 1 images or less-noisy Year 10 images.

The development notebook for deterministic CNNs trained on Y1 images can be found in `LSST-CNN-noisy.ipynb`, and CNNs trained on Y10 images can be found in `LSST-CNN-pristine.ipynb`.

The development notebook for fully probabilistic BNNs trained on Y1 images can be found in `LSST-BNN-noisy.ipynb`.

High-performance computing was performed using the [Fermilab Elastic Analysis Facility](https://eafjupyter.readthedocs.io/en/latest/).

Plots for various models and exploratory data analysis can be found in the `Plots` directory. Completed models can be found in the `Models` directory.

## Authors
- [Marina M. Dunn](https://marinadunn.github.io) (<mdunn014@ucr.edu>)

Deep Skies Lab Research Mentors:
- Dr. Aleksandra Ćiprijanović (<aleksand@fnal.gov>)
- Dr. Brian Nord (<nord@fnal.gov>)

## Acknowledgements
We acknowledge the Deep Skies Lab as a community of multi-domain experts and collaborators who’ve facilitated an environment of open discussion, idea-generation, and collaboration. This community was important for the development of this project.

## References
DeepAdversaries paper: [arXiv:2111.00961](https://ui.adsabs.harvard.edu/abs/2021arXiv211100961C/abstract)

DeepAdversaries original dataset: [DeepAdversaries Data](https://zenodo.org/record/5514180#.ZERSvS_MJp8)
