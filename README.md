# LSST Galaxy Morphology Classification Using Bayesian Neural Networks
This repository contains development code and documentation for my M.S. Engineering: Data Science thesis research (final project expected June 2023) at UC Riverside, entitled "Galaxy Morphology Classification Using Bayesian Neural Networks for the Legacy Survey of Space and Time (LSST)." UCR Project Advisor: Dr. Bahram Mobasher, Deep Skies Lab Research Advisors: Dr. Aleksandra Ciprijanovic (Fermilab, U Chicago), and Dr. Brian Nord (Fermilab, U Chicago, Kavli Institute for Cosmological Physics, MIT)

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

## Development
The development and training notebook for standard Convlolutional Neural Networks, training with Year 1 (noisy) data, can be found in:
```
LSST-CNN-noisy.ipynb
```
The development and training notebook for standard Convlolutional Neural Networks, training with Year 10 (low-noise) data can be found in:
```
LSST-CNN-pristine.ipynb
```
The development and training notebook for fully probabilistic Bayesian Neural Network, training with Year 1 (noisy) data, was can be found in:
```
LSST-BNN-noisy.ipynb
```
This was completed using the Fermilab Elastic Analysis Facility.

Plots for various models and exploratory data analysis can be found in the folder ```Plots```.

## Requirements
This code was developed using Python 3.10, Tensorflow-Probability 0.16.0, and Tensorflow 2.9.0. It is recommended to create a new Python virtual environment before running this code. The file `requirements.txt` lists all the Python libraries needed to run the notebooks; they can be installed by running the command:
```
python3 -m pip install -r requirements.txt
```

## Authors
- [Marina M. Dunn](https://orcid.org/0000-0001-5374-1644) (University of California, Riverside/NASA Goddard Space Flight Center, <mdunn014@ucr.edu>)
- [Dr. Aleksandra Ciprijanovic](https://orcid.org/0000-0003-1281-7192) (Fermi National Accelerator Lab/University of Chicago, <aleksand@fnal.gov>)

## Acknowledgements
We acknowledge the Deep Skies Lab as a community of multi-domain experts and collaborators who’ve facilitated an environment of open discussion, idea-generation, and collaboration. This community was important for the development of this project.

## References
DeepAdversaries paper: [arXiv:2111.00961](https://ui.adsabs.harvard.edu/abs/2021arXiv211100961C/abstract)

DeepAdversaries original dataset: [DeepAdversaries Data](https://zenodo.org/record/5514180#.ZERSvS_MJp8)
