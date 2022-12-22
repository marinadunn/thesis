# Thesis
This repository contains relevant development code and documentation for my M.S. Engineering: Data Science thesis (expected June 2023) research at UC Riverside, entitled "Classifying Galaxy Morphologies for LSST Using Bayesian Neural Networks."

## About: 
The goal of this project is to first develop a Convolutional Neural Network (CNN) model, and eventually a Bayesian Neural Network (BNN) model, that is trained on a simulated mock catalog of galaxies created from the Illustris simulation for the Vera C. Rubin Observatory’s Legacy Survey of Space and Time (LSST), and accurately classify their morphologies as either 'spiral', 'elliptical', or 'merger.' This mock catalog is designed to resemble LSST observations at years 1 (where data is noisier) and 10 (where data is less-noisy).

## Development Steps:
- Pre-processing data and Exploratory Data Analysis
- Develop deterministic Convolutional Neural Network model trained on noisy Year 1 data
- Evaluate CNN model performance on less-noisy Year 10 (expected to have poor performance)
- Perform Transfer Learning using these results, and re-test
- Develop a Bayesian Neural Network model
- Evaluate BNN model on Year 10 data

## Data
The simulated mock LSST catalog used was created from the [Illustris TNG100 simulation](https://www.illustris-project.org) as part of the [DeepAdversaries project](https://github.com/AleksCipri/DeepAdversaries), with observational noise and PSF added using the [GalSim package](https://github.com/GalSim-developers/GalSim). Both the original raw data files, as well as smaller subsets, can be accessed on the [DeepAdversaries Zenodo page](https://zenodo.org/record/5514180#.Ymb3zi-B2L2) in .npy file format. The mock image labels are one-hot encoded into 3 classes: Spiral ('0'), Elliptical ('1'), or Merger ('2’).

Data Distribution:
* Training set: 23,487 images (10,017 Spirals, 5,705 Ellipticals, 7,765 Mergers)
* Testing set: 6,715 images (2,863 Spirals; 1,631 Ellipticals; 2,221 Mergers)
* Validation set: 3,355 images (1,432 Spirals; 815 Ellipticals; 1,108 Mergers)
* Total: 33,557 images (Splitting fraction 70% train : 20% test : 10% validation)

Example Images:
![Example Images of Year 1 and Year 10 spiral galaxies, elliptical galaxies, and galaxy mergers](https://github.com/marinadunn/thesis/blob/main/Plots/example_images.jpg "Example Images of Year 1 & Year 10 galaxies")

## Training
Model development and training on year 1 (noisy) data can be found in the notebook
```
LSST-BNN-noisy.ipynb
```
Model development and training on year 10 (low-noise) data can be found in the notebook
```
LSST-BNN-pristine.ipynb
```

## Requirements
This code was developed using Python 3.10 and TensorFlow 2.9.0. It is recommended to create a new Python virtual environment before running this code. The file `requirements.txt` lists all the Python libraries needed to run the notebooks; they can be installed by running the command:
```
pip install -r requirements.txt
```

## Authors
- [Marina M. Dunn](https://orcid.org/0000-0001-5374-1644) (UC Riverside, <mdunn014@ucr.edu>)
- [Dr. Aleksandra Ciprijanovic](https://orcid.org/0000-0003-1281-7192) (Fermi National Accelerator Lab, <aleksand@fnal.gov>)

## References
DeepAdversaries paper: [arXiv:2111.00961](https://ui.adsabs.harvard.edu/abs/2021arXiv211100961C/abstract)
