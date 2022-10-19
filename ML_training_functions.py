## Functions for ML Training

# Arrays
import numpy as np
from numpy import arcsinh as arcsinh
import pandas as pd

# System
import os
import sys
import json

# Plotting
import matplotlib.pyplot as plt

# ML - building and training CNN
import tensorflow_probability as tfp
import tensorflow as tf

from tensorflow.keras.models import Model, model_from_json, load_model

# evaluating CNN and hyperparameter optimization
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, auc


# Add-ons
# from tensorflow_addons.optimizers import CyclicalLearningRate, AdamW

## Pre-processing

# Function for viewing pixel values by filter color
def plot_filters(t):
    
    plt.hist(t[:, 0].numpy().ravel(), bins=30, color = 'blue', alpha = 0.7, density=True, align='mid', stacked=True)
    plt.hist(t[:, 1].numpy().ravel(), bins=30, color = 'red', alpha = 0.7, density=True, align='mid', stacked=True)
    plt.hist(t[:, 2].numpy().ravel(), bins=30, color = 'green', alpha = 0.7, density=True, align='mid', stacked=True)
    
    plt.xlabel("Pixel Values")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of Pixels")
    plt.tight_layout()

    print('Min: %.3f, Max: %.3f' % (np.amin(t[:, 0]), np.amax(t[:, 0])))
    print('Min: %.3f, Max: %.3f' % (np.amin(t[:, 1]), np.amax(t[:, 1])))
    print('Min: %.3f, Max: %.3f' % (np.amin(t[:, 2]), np.amax(t[:, 2])))

# Function for scaling pixel values
def scale_pixels(t):
    # clip outliers based on global values
    global_min = np.percentile(t, 0.1)
    global_max = np.percentile(t, 99.9)
    
    # for each color filter
    for i in range(0, 3):
        c = .85/global_max
        t[:,i] = np.clip(t[:,i].numpy(), global_min, global_max)
        # get pixel values as close to 0-1 as possible
        t[:,i] = t[:,i].numpy().arcsinh(c * t[:, i])
        t[:,i] = (t[:,i].numpy() + 1.0) / 2.0    
        
# Function for calculating mean and standard deviation for pixel values
def mean_std(t):
    
    mean1 = t[:,0].mean().item()
    mean2 = t[:,1].mean().item()
    mean3 = t[:,2].mean().item()
    mean = [mean1, mean2, mean3]

    std1 = t[:,0].std().item()
    std2 = t[:,1].std().item()
    std3 = t[:,2].std().item()
    std = [std1, std2, std3]

    return mean, std

## Training

# Function for generating class weights
def generate_class_weights(class_series):
    # class is one-hot encoded, so transform to categorical labels to use compute_class_weight   
    class_series = np.argmax(class_series, axis=1)
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    
    return dict(zip(class_labels, class_weights))

# Function for compiling model
def compile_model(model, loss, optimizer):
    metrics = ['accuracy']
    model.compile(optimizer = optimizer,loss = loss, metrics = metrics)
    model.summary()
    
# Function for visualizing training history
def plot_training(model, history):
    
    # Function to plot the Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Function to plot the Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = list(range(len(loss)))
    
    ### Plotting
    # plot accuracy
    figsize=(6, 4)
    figure = plt.figure(figsize=figsize)
    plt.plot(epochs, acc, 'navy', label='Accuracy')
    plt.plot(epochs, val_acc, 'deepskyblue', label= "Validation Accuracy")    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{model.name} Accuracy Training History")
    plt.legend(loc='best')
    #plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(fname=f"{model.name} Accuracy Training History", format='png')
    plt.show()

    # plot loss
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

# Function for making classification report
def make_classification_report(y_data, y_pred):
    classification_metrics = classification_report(y_data, y_pred, target_names=class_names)
    return classification_metrics

# Function for making confusion matrix
def make_cm(model, y_data, y_pred):
    """
    Given a keras model, true labels, and predicted labels, create a confusion 
    matrix (cm), and make a sklearn Confusion Matrix visualization for plotting.
    
    Arguments
    ---------
    """
    labels = [0, 1, 2]
    cm = confusion_matrix(y_data, y_pred, labels=labels)
    cm = cm.astype('float')
    cd = ConfusionMatrixDisplay(cm, display_labels=class_names)
    return cd

# Function for saving model data
def save_model_data(model, history):
    model.save(filepath = f'{model.name}.json', include_optimizer = True, overwrite = True)
    
    # Saving history as .npy file for future use
    np.save(f'{model.name}_history.npy', history.history)