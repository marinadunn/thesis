import os, sys
import argparse
import time
import random
import numpy as np
from numpy import arcsinh as arcsinh
import pandas as pd
import matplotlib.pyplot as plt

# check data sizes are correct
def check_data_sizes(x_train, x_test, x_valid):
    NUM_TRAIN = 23487
    NUM_TEST = 6715
    NUM_VALIDATION = 3355
    NUM_TOTAL = NUM_TRAIN + NUM_TEST + NUM_VALIDATION
    print(NUM_TOTAL)
    assert NUM_TOTAL == len(x_train) + len(x_test) + len(x_valid), "total training, test, validation samples not equal to total samples - exiting"

# Function for plotting histogram of all datasets by label
def labels_hist(y_train, y_test, y_validation):

    # create dataframes of labels
    Y_train_df = pd.DataFrame(y_train)
    Y_test_df = pd.DataFrame(y_test)
    Y_valid_df = pd.DataFrame(y_validation)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 6))

    ax1.hist(Y_train_df[0].loc[Y_train_df[0] == 1.], 
            label='Spirals', histtype='bar', color='teal')
    ax1.hist(Y_train_df[2].loc[Y_train_df[2] == 1.], 
            label='Mergers', histtype='bar', color='pink', alpha = 0.9)
    ax1.hist(Y_train_df[1].loc[Y_train_df[1] == 1.], 
            label='Ellipticals', histtype='bar', color='purple')
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(0.9, 1.2)
    ax1.set_ylim(0, 11000)
    ax1.set_title('Training Set')

    ax2.hist(Y_test_df[0].loc[Y_test_df[0] == 1.], 
            label='Spirals', histtype='bar', color='teal')
    ax2.hist(Y_test_df[2].loc[Y_test_df[2] == 1.], 
            label='Mergers', histtype='bar', color='pink', alpha = 0.9)  
    ax2.hist(Y_test_df[1].loc[Y_test_df[1] == 1.],
            label='Ellipticals', histtype='bar', color='purple')
    ax2.get_xaxis().set_visible(False)
    ax2.set_xlim(0.9, 1.2)
    ax2.set_ylim(0, 11000)
    ax2.set_title('Test Set')

    ax3.hist(Y_valid_df[0].loc[Y_valid_df[0] == 1.], 
            label='Spirals', histtype='bar', color='teal')
    ax3.hist(Y_valid_df[2].loc[Y_valid_df[2] == 1.], 
            label='Mergers', histtype='bar', color='pink', alpha = 0.9)
    ax3.hist(Y_valid_df[1].loc[Y_valid_df[1] == 1.], 
            label='Ellipticals', histtype='bar', color='purple')
    ax3.get_xaxis().set_visible(False)
    ax3.set_xlim(0.9, 1.2)
    ax3.set_ylim(0, 11000)
    ax3.set_title('Validation Set')

    plt.tight_layout()
    plt.legend()
    plt.savefig("Labels by Dataset")
    plt.show()

# Function for histogram of pixel intensities for all datasets by filter
def plot_filters(x_train, x_test, x_validation, year, scaled=False):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7), sharex=True, sharey=True)

    if scaled == True:
        range = [0, 1]
        title = f"Y{year} Data Scaled Pixel Intensities"
    else:
        range = [-10000, 10000]
        title = f"Y{year} Raw Data Pixel Intensities"

    # Training set
    ax1.hist(x_train[:, 0, :, :].ravel(), bins=100, color = 'g', alpha = 0.7,
             density=True, range=range, stacked=True)
    ax1.hist(x_train[:, 1, :, :].ravel(), bins=100, color = 'r', alpha = 0.5, 
            density=True, range=range, stacked=True)
    ax1.hist(x_train[:, 2, :, :].ravel(), bins=100, color = (0.3816778909618176, 0, 0), 
             alpha = 0.4, density=True, range=range, stacked=True)
    ax1.set_title("Train")

    # Test set
    ax2.hist(x_test[:, 0, :, :].ravel(), bins=100, color = 'g', alpha = 0.7, 
            density=True, range=range, stacked=True)
    ax2.hist(x_test[:, 1, :, :].ravel(), bins=100, color = 'r', alpha = 0.5, 
            density=True, range=range, stacked=True)
    ax2.hist(x_test[:, 2, :, :].ravel(), bins=100, color = (0.3816778909618176, 0, 0),
             alpha = 0.4, density=True, range=range, stacked=True)
    ax2.set_title("Test")

    # Validation set
    ax3.hist(x_validation[:, 0, :, :].ravel(), bins=100, color = 'g', alpha = 0.7, 
            density=True, range=range, stacked=True)
    ax3.hist(x_validation[:, 1, :, :].ravel(), bins=100, color = 'r', alpha = 0.5, 
            density=True, range=range, stacked=True)
    ax3.hist(x_validation[:, 2, :, :].ravel(), bins=100, color = (0.3816778909618176, 0, 0), 
             alpha = 0.4, density=True, range=range, stacked=True)
    ax3.set_title("Validation")

    fig.suptitle(title, fontsize=14)
    fig.supxlabel("Pixel Intensity")
    fig.supylabel("Number of Pixels")

    plt.tight_layout()
    plt.legend(['G (464 nm)', 'R (658 nm)', 'I (806 nm)'])
    plt.savefig(title)
    plt.show()

    print(f'Y{year} Train:\n')
    print('G Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 0, :, :]), np.amax(x_train[:, 0, :, :])))
    print('R Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 1, :, :]), np.amax(x_train[:, 1, :, :])))
    print('I Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 2, :, :]), np.amax(x_train[:, 2, :, :])))

    print(f'\nY{year} Test:\n')
    print('G Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 0, :, :]), np.amax(x_test[:, 0, :, :])))
    print('R Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 1, :, :]), np.amax(x_test[:, 1, :, :])))
    print('I Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 2, :, :]), np.amax(x_test[:, 2, :, :])))

    print(f'\nY{year} Validation:\n')
    print('G Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 0, :, :]), 
                                      np.amax(x_validation[:, 0, :, :])))
    print('R Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 1, :, :]), 
                                      np.amax(x_validation[:, 1, :, :])))
    print('I Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 2, :, :]), 
                                      np.amax(x_validation[:, 2, :, :])))

# Function for scaling pixel values
def update_sinh(x_data):
    # clip outliers based on global values
    global_min = np.percentile(x_data, 0.1)
    global_max = np.percentile(x_data, 99.9)

    # for each color filter
    for i in range(0, 3):
        #g, r, i
        c = .85 / global_max
        # gets you close to arcsinh(max_x) = 1, arcsinh(min_x) = 0
        x_data[:, i, :, :] = np.clip(x_data[:, i, :, :], global_min, global_max)
        x_data[:, i, :, :] = arcsinh(c * x_data[:, i, :, :])
        x_data[:, i, :, :] = (x_data[:, i, :, :] + 1.0) / 2.0

# Function for calculating mean and standard deviation of pixel intensities per filter
def mean_std(x_data):

    mean1 = np.mean(x_data[:, 0, :, :])
    mean2 = np.mean(x_data[:, 1, :, :])
    mean3 = np.mean(x_data[:, 2, :, :])
    mean = [mean1, mean2, mean3]

    std1 = np.std(x_data[:, 0, :, :])
    std2 = np.std(x_data[:, 1, :, :])
    std3 = np.std(x_data[:, 2, :, :])
    stdev = [std1, std2, std3]

    print("G, R, I mean: ", mean)
    print("G, R, I standard deviation: ", stdev)

    return mean, stdev

# plot example images from training set
def plot_examples(x_train, i, year):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), 
                                        constrained_layout=True)
    if 0 < i <= 10018:
        label = 'Spiral'
    elif 10018 < i <= 15723:
        label = 'Elliptical'
    elif 15723 < i:
        label = 'Merger'
        
    fig.suptitle(f'Y{year} {label} Galaxies', y=0.9, fontsize=14)

    ax1.imshow(x_train[i, 0, :, :])
    ax1.axis("off")
    ax1.set_title("Green (G)")
    
    ax2.imshow(x_train[i, 1, :, :])
    ax2.axis("off")
    ax2.set_title("Red (R)")

    ax3.imshow(x_train[i, 2, :, :])
    ax3.axis("off")
    ax3.set_title("Near-Infrared (I)")
    
    plt.savefig(f'Y{year} {label} Galaxies Example Images')
    plt.show()