## Functions for ML Training

# Arrays
import numpy as np
from numpy import arcsinh as arcsinh
import pandas as pd

# System
import os

# Plotting
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ML - building and training CNN
import tensorflow_probability as tfp
import tensorflow as tf

# evaluating CNN and hyperparameter optimization
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, roc_auc_score, auc, log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

## Pre-processing

# Function for plotting histogram of all datasets by label
def labels_hist(y_train, y_test, y_validation):
    
    # create dataframes of labels
    Y_train_df = pd.DataFrame(y_train)
    Y_test_df = pd.DataFrame(y_test)
    Y_valid_df = pd.DataFrame(y_validation)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 4))

    ax1.hist(Y_train_df[0].loc[Y_train_df[0] == 1.], label='Spirals', histtype='bar', color='teal')
    ax1.hist(Y_train_df[2].loc[Y_train_df[2] == 1.], label='Mergers', histtype='bar', color='pink', alpha = 0.9)
    ax1.hist(Y_train_df[1].loc[Y_train_df[1] == 1.], label='Ellipticals', histtype='bar', color='purple')
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(0.9,1.2)
    ax1.set_ylim(0,11000)
    ax1.set_title('Training Set')

    ax2.hist(Y_test_df[0].loc[Y_test_df[0] == 1.], label='Spirals', histtype='bar', color='teal')
    ax2.hist(Y_test_df[2].loc[Y_test_df[2] == 1.], label='Mergers', histtype='bar', color='pink', alpha = 0.9)  
    ax2.hist(Y_test_df[1].loc[Y_test_df[1] == 1.],label='Ellipticals', histtype='bar', color='purple')
    ax2.get_xaxis().set_visible(False)
    ax2.set_xlim(0.9,1.2)
    ax2.set_ylim(0,11000)
    ax2.set_title('Test Set')

    ax3.hist(Y_valid_df[0].loc[Y_valid_df[0] == 1.], label='Spirals', histtype='bar', color='teal')
    ax3.hist(Y_valid_df[2].loc[Y_valid_df[2] == 1.], label='Mergers', histtype='bar', color='pink', alpha = 0.9)
    ax3.hist(Y_valid_df[1].loc[Y_valid_df[1] == 1.],label='Ellipticals', histtype='bar',color='purple')
    ax3.get_xaxis().set_visible(False)
    ax3.set_xlim(0.9,1.2)
    ax3.set_ylim(0,11000)
    ax3.set_title('Validation Set')

    plt.tight_layout()
    plt.legend()
    plt.savefig(fname=f"Dataset by label", format='jpg')
    plt.show()

# Function for viewing pixel values for all datasets by filter color
def plot_filters(x_train, x_test, x_validation):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)

    # Training set
    ax1.hist(x_train[:, 0].ravel(), bins=30, color = 'blue', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax1.hist(x_train[:, 1].ravel(), bins=30, color = 'red', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax1.hist(x_train[:, 2].ravel(), bins=30, color = 'green', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax1.set_title("Train")

    # Test set
    ax2.hist(x_test[:, 0].ravel(), bins=30, color = 'blue', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax2.hist(x_test[:, 1].ravel(), bins=30, color = 'red', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax2.hist(x_test[:, 2].ravel(), bins=30, color = 'green', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax2.set_title("Test")

    # Validation set
    ax3.hist(x_validation[:, 0].ravel(), bins=30, color = 'blue', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax3.hist(x_validation[:, 1].ravel(), bins=30, color = 'red', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax3.hist(x_validation[:, 2].ravel(), bins=30, color = 'green', alpha = 0.2, density=True, range=[-10000, 10000], stacked=True)
    ax3.set_title("Validation")

    fig.suptitle("Distribution of Pixels")
    fig.supxlabel("Pixel Values")
    fig.supylabel("Relative Frequency")
    plt.tight_layout()
    plt.show()

    print('Train:\n')
    print('Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 0]), np.amax(x_train[:, 0])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 1]), np.amax(x_train[:, 1])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 2]), np.amax(x_train[:, 2])))

    print('\nTest:\n')
    print('Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 0]), np.amax(x_test[:, 0])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 1]), np.amax(x_test[:, 1])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 2]), np.amax(x_test[:, 2])))

    print('\nValidation:\n')
    print('Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 0]), np.amax(x_validation[:, 0])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 1]), np.amax(x_validation[:, 1])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 2]), np.amax(x_validation[:, 2])))

# Function for scaling pixel values
def scale_pixels(x_data):
    # clip outliers based on global values
    global_min = np.percentile(x_data, 0.1)
    global_max = np.percentile(x_data, 99.9)
    
    # for each color filter
    for i in range(0, 3):
        #g, r, i
        c = .85 / global_max
        # gets you close to arcsinh(max_x) = 1, arcsinh(min_x) = 0
        x_data[:, i] = np.clip(x_data[:,i], global_min, global_max)
        x_data[:, i] = arcsinh(c * x_data[:, i])
        x_data[:, i] = (x_data[:, i] + 1.0) / 2.0

# Function for viewing pixel values for all datasets by filter color after scaling
def plot_filters_scaled(x_train, x_test, x_validation):
        
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)

    # Training set
    ax1.hist(x_train[:, 0].ravel(), bins=30, color = 'blue', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax1.hist(x_train[:, 1].ravel(), bins=30, color = 'red', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax1.hist(x_train[:, 2].ravel(), bins=30, color = 'green', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax1.set_title("Train")

    # Test set
    ax2.hist(x_test[:, 0].ravel(), bins=30, color = 'blue', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax2.hist(x_test[:, 1].ravel(), bins=30, color = 'red', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax2.hist(x_test[:, 2].ravel(), bins=30, color = 'green', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax2.set_title("Test")

    # Validation set
    ax3.hist(x_validation[:, 0].ravel(), bins=30, color = 'blue', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax3.hist(x_validation[:, 1].ravel(), bins=30, color = 'red', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax3.hist(x_validation[:, 2].ravel(), bins=30, color = 'green', alpha = 0.2, density=True, range=[0, 1], stacked=True)
    ax3.set_title("Validation")

    fig.suptitle("Distribution of Pixels (Scaled)")
    fig.supxlabel("Scaled Pixel Values")
    fig.supylabel("Relative Frequency")
    plt.tight_layout()
    plt.show()

    print('Train Scaled:\n')
    print('Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 0]), np.amax(x_train[:, 0])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 1]), np.amax(x_train[:, 1])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_train[:, 2]), np.amax(x_train[:, 2])))

    print('\nTest Scaled:\n')
    print('Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 0]), np.amax(x_test[:, 0])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 1]), np.amax(x_test[:, 1])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_test[:, 2]), np.amax(x_test[:, 2])))

    print('\nValidation Scaled:\n')
    print('Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 0]), np.amax(x_validation[:, 0])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 1]), np.amax(x_validation[:, 1])))
    print('Min: %.3f, Max: %.3f' % (np.amin(x_validation[:, 2]), np.amax(x_validation[:, 2])))
        
# Function for calculating mean and standard deviation of pixel values
def mean_std(x_data):
    
    mean1 = x_data[:, 0].mean().item()
    mean2 = x_data[:, 1].mean().item()
    mean3 = x_data[:, 2].mean().item()
    mean = [mean1, mean2, mean3]

    std1 = x_data[:, 0].std().item()
    std2 = x_data[:, 1].std().item()
    std3 = x_data[:, 2].std().item()
    std = [std1, std2, std3]

    return mean, std

## Training

# Function for generating class weights
def generate_class_weights(class_series):
    # class is one-hot encoded, so transform to categorical labels to use compute_class_weight   
    class_series = np.argmax(class_series, axis=1)
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=class_labels, 
                                         y=class_series)
    class_weights = dict(zip(class_labels, class_weights))
    return class_weights
    
# Function for visualizing training history
def plot_training(model, history):
    
    # Function to plot the Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Function to plot the Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = list(range(len(loss)))
    
    # Plot accuracy
    figsize=(6, 4)
    figure = plt.figure(figsize=figsize)
    plt.plot(epochs, acc, 'navy', label='Accuracy')
    plt.plot(epochs, val_acc, 'deepskyblue', label= "Validation Accuracy")    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{model.name} Accuracy Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname=f"{model.name} Accuracy Training History", format='png')
    plt.show()

    # Plot loss
    figsize=(6, 4)
    figure = plt.figure(figsize=figsize)
    plt.plot(epochs, loss, 'red', label='Loss')
    plt.plot(epochs, val_loss, 'lightsalmon', label= "Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{model.name} Loss Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname=f"{model.name} Loss Training History", format='png')
    plt.show()
    
# Function for evaluating model
def evaluate_model(model, x_data, y_data):
    score = model.evaluate(x_data, y_data, verbose=True)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Function for making confusion matrix
def plot_cm(model, y_data, y_pred, class_names=class_names):
    """
    Given a keras model, true target labels, and predicted labels, create a confusion 
    matrix (cm), and visualize.
    
    Arguments
    ---------
    """
    class_names = ['Spiral', 'Elliptical', 'Merger']
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels = labels)
    cm = cm.astype('float')
    cd = ConfusionMatrixDisplay(cm, display_labels=class_names)
    return cd

def plot_roc(y_data, y_pred, class_names=class_names):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    NUM_CLASSES = len(class_names)
    
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], thresholds = roc_curve(y_data[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize = (6, 5), tight_layout=True)
    colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']
    for i, color, lbl in zip(range(NUM_CLASSES), colors, class_names):
        plt.plot(fpr[i], tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.3f})'.format(lbl, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Selectivity)')
    plt.legend(loc="lower right")
    plt.title(f'{model.name} ROC Curves')
    plt.savefig(fname=f"{model.name}_ROC", format='png')
    plt.show()
    
def plot_class_activation_maps(model, x_data):

    # Extract outputs of all layers except the input layer
    layer_outputs = [layer.output for layer in model.layers[1:]]

    # Create model that will return these outputs, given the model input
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 

    # returns the values of the layer activations in model
    # Returns a list of Numpy arrays: one array per layer activation
    activations = activation_model.predict(x_data) 
    
    # plot activation maps for all filters in each convolutional layer
    
    fig1 = plt.figure(figsize=(11, 1.5))

    for i in range(8):
        plt.subplot(1, 8, i + 1)
        layer_activation = activations[0]
        plt.imshow(layer_activation[1649, i, :, :], cmap='viridis', aspect='auto')
        plt.axis("off")
        plt.subplots_adjust(hspace=0, wspace=0)


    fig2 = plt.figure(figsize=(11, 3))  
    for i in range(16):
        plt.subplot(2, 16, i + 1)
        layer_activation = activations[4]
        plt.imshow(layer_activation[1649, i, :, :], cmap='viridis', aspect='auto')
        plt.axis("off")
        plt.subplots_adjust(hspace=0, wspace=0)


    fig3 = plt.figure(figsize=(15, 6))  
    for i in range(32):
        plt.subplot(4, 32, i + 1)
        layer_activation = activations[8]
        plt.imshow(layer_activation[1649, i, :, :], cmap='viridis', aspect='auto')
        plt.axis("off")
        plt.subplots_adjust(hspace=0, wspace=0)