# Arrays
import numpy as np
import pandas as pd

# System
import os
import datetime
import math
import random

# Plotting
import matplotlib
import matplotlib.pyplot as plt
# sklearn metrics and plotting
import sklearn
from sklearn.metrics import (roc_curve, roc_auc_score, auc, log_loss,
                             precision_score, recall_score, f1_score, 
                             accuracy_score, classification_report, 
                             ConfusionMatrixDisplay, confusion_matrix)
# evaluating CNN and hyperparameter optimization
from sklearn.utils.class_weight import compute_class_weight

# Function for generating class weights
def generate_class_weights(labels):
    # class is one-hot encoded, so transform to categorical labels to use compute_class_weight   
    class_series = np.argmax(labels, axis=1)
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight = 'balanced', 
                                         classes = class_labels, 
                                         y = class_series)
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
    plt.plot(epochs, val_acc, 'deepskyblue', label="Validation Accuracy")    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{model.name} Accuracy Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{model.name} Accuracy Training History")
    plt.show()

    # Plot loss
    figsize=(6, 4)
    figure = plt.figure(figsize=figsize)
    plt.plot(epochs, loss, 'red', label='Loss')
    plt.plot(epochs, val_loss, 'lightsalmon', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{model.name} Loss Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{model.name} Loss Training History")
    plt.show()
    
# Function for evaluating model
def evaluate_model(model, x_data, y_data):
    score = model.evaluate(x_data, y_data, verbose=True)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Function for making confusion matrix
def plot_cm(model, y_data, y_pred):
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
    cd = ConfusionMatrixDisplay(cm, display_labels = class_names)
    return cd

def plot_roc(y_data, y_pred, model):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_names = ['Spiral', 'Elliptical', 'Merger']
    NUM_CLASSES = len(class_names)
    
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], thresholds = roc_curve(y_data[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize = (6, 5), tight_layout=True)
    colors = ['blue', 'red', 'green', 'brown', 'purple', 
              'pink', 'orange', 'black', 'yellow', 'cyan']
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
    plt.savefig(f"{model.name}_ROC")
    plt.show()