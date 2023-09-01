import random
import numpy as np
import pandas as pd
np.set_printoptions(edgeitems=25, linewidth=100000)
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 10000000

# ML
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import (Callback, ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, LambdaCallback, LearningRateScheduler)
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, ZeroPadding2D, ReLU,
                                     AveragePooling2D, Add, Conv2D, MaxPool2D, BatchNormalization,
                                     Input, Flatten, Dense, Dropout)
from tensorflow.keras.initializers import VarianceScaling, RandomUniform, HeNormal

# tensorflow add-ons
import tensorflow_addons as tfa

# tensorflow probability
import tensorflow_probability as tfp
tfd = tfp.distributions

# sklearn metrics and plotting
from sklearn.metrics import (roc_curve, roc_auc_score, auc, log_loss)

## Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Fixing global random seed for reproducibility
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


def analyse_model_prediction(data, true_labels,
                             model, image_num,
                             run_ensemble=False,
                             NUM_CLASSES=NUM_CLASSES):
    if run_ensemble:
        ensemble_size = 200
    else:
        ensemble_size = 1
    image = data[image_num]
    true_label = true_labels[image_num]
    predicted_probabilities = np.empty(shape=(ensemble_size, NUM_CLASSES))

    for i in range(ensemble_size):
        predicted_probabilities[i] = model(image[np.newaxis, :]).numpy().mean()[0]
    model_prediction = model(image[np.newaxis, :])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 2),
                                   gridspec_kw={'width_ratios': [2, 4]})

    # Show the image and the true label
    ax1.imshow(image[...][0])
    ax1.axis('off')
    if true_label[0] == 1:
        ax1.set_title('True label: Spiral')
    elif true_label[1] == 1:
        ax1.set_title('True label: Elliptical')
    elif true_label[2] == 1:
        ax1.set_title('True label: Merger')

    # Show a 95% prediction interval of model predicted probabilities
    pct_2p5 = np.array([np.percentile(predicted_probabilities[:, i], 2.5) for i in range(NUM_CLASSES)])
    pct_97p5 = np.array([np.percentile(predicted_probabilities[:, i], 97.5) for i in range(NUM_CLASSES)])
    bar = ax2.bar(np.arange(NUM_CLASSES), color='red')
    if true_label[0] == 1:
        bar[0].set_color('green')
    elif true_label[1] == 1:
        bar[1].set_color('green')
    elif true_label[2] == 1:
        bar[2].set_color('green')
    ax2.bar(np.arange(NUM_CLASSES), color='white', linewidth=1, edgecolor='white')
    ax2.set_xticks(np.arange(NUM_CLASSES))
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Probability')
    ax2.set_title('Example Model Estimated Probabilities')
    plt.savefig(f'{str(model)}_estimated_probabilities_examples.png', dpi=300)
    plt.show()


def plot_acc_loss(history, model, model_name=None):
    """Plot model accuracy and loss training history."""

    if model_name is None:
        model_name = model.name
    else:
        model_name = model_name

    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']
    epochs = list(range(len(loss)))

    # Plot accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, acc, 'navy', label='Accuracy')
    plt.plot(epochs, val_acc, 'deepskyblue', label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{model_name} Accuracy Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"Plots/{model_name} Accuracy Training History", dpi=300)
    plt.show()

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, 'red', label='Loss')
    plt.plot(epochs, val_loss, 'lightsalmon', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{model_name} Loss Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"Plots/{model_name} Loss Training History", dpi=300)
    plt.show()


def plot_roc(model, y_pred, Y_test, NUM_CLASSES, model_name=None):
    """Plot ROC curves & AUC for a categorical classifier."""

    if model_name is None:
        model_name = model.name
    else:
        model_name = model_name

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], thresholds = roc_curve(Y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6, 5), tight_layout=True)
    colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']
    for i, color, lbl in zip(range(NUM_CLASSES), colors, class_names):
        plt.plot(fpr[i], tpr[i], color = color, lw = 1.5,
        label='ROC Curve of class {0} (area = {1:0.3f})'.format(lbl, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Selectivity)')
    plt.legend(loc="lower right")
    plt.title(f'{model_name} Test ROC Curves')
    plt.tight_layout()
    plt.savefig(f'Plots/{model_name} ROC', dpi=300)
    plt.show()