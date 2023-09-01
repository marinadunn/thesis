# Import user-defined functions
from preprocessing import *
from training import *

import datetime
import math
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(edgeitems=25, linewidth=100000)
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 10000000
from numpy import arcsinh as arcsinh

# ML
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
# data augmentation
# from tensorflow.keras.layers import RandomFlip, RandomZoom, RandomRotation, ReLU
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
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
from tensorflow_addons.optimizers import CyclicalLearningRate, AdamW, ExponentialCyclicalLearningRate

# tensorflow probability
import tensorflow_probability as tfp
tfd = tfp.distributions

## Plotting
import visualkeras
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import ImageFont
import scipy.stats

NUM_CLASSES = 3

# CNN

# Define probabilistic CNN model using ResNet18 architecture
def conv3x3(x, out_planes, stride=1):
    x = ZeroPadding2D(padding=1)(x)
    return Conv2D(filters=out_planes, kernel_size=3, strides=stride)(x)


# Residual block
def basic_block(x, planes, stride=1, downsample=None):
    # Original input
    identity = x

    # First convolutional layer per block
    out = conv3x3(x, planes, stride=stride)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Second convolutional layer per block
    out = conv3x3(out, planes)
    out = BatchNormalization()(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = tf.keras.layers.add([identity, out])
    out = Activation('relu')(out)

    return out


def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    # Do downsample if stride not 1
    if stride != 1 or inplanes != planes:
        # Use kernel size 1, stride 2 for convolutional layers in residual blocks 2 and after
        downsample = [
            Conv2D(filters=planes, kernel_size=1, strides=stride),
            BatchNormalization(),
        ]
    # Create residual block; blocks 2 and after will have downsample
    x = basic_block(x, planes, stride, downsample)

    # Create 2 residual blocks per module
    for i in range(1, blocks):
        x = basic_block(x, planes)

    return x


def resnet18_prob(x, num_classes=NUM_CLASSES):
    # First convolutional layer is 7x7 with 64 output channels and stride 2, then Max-pooling layer with stride 2
    x = ZeroPadding2D(padding=3)(x)
    x = Conv2D(filters=64, kernel_size=7, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=1)(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # ResNet18 uses two residual blocks for each module (filter size) here, 4 convolutional layers per module
    # First 2 blocks use stride 1
    x = make_layer(x, 64, 2)
    # All layers after this use stride 2
    x = make_layer(x, 128, 2, stride=2)
    x = make_layer(x, 256, 2, stride=2)
    x = make_layer(x, 512, 2, stride=2)

    # Final layers
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    distribution_params = Dense(tfp.layers.OneHotCategorical.params_size(num_classes), activation='relu')(x)
    outputs = tfp.layers.OneHotCategorical(num_classes,
                                           convert_to_tensor_fn=tfp.distributions.Distribution.mode)(distribution_params)

    return outputs

# BNN


def conv3x3_bnn(x, out_planes, stride=1, name=None):
    x = ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return tfp.layers.Convolution2DReparameterization(filters=out_planes,
                                                        kernel_size=3,
                                                        strides=stride,
                                                        kernel_divergence_fn=kl_divergence_function,
                                                        name=name)(x)


def basic_block_bnn(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3_bnn(x, planes, stride=stride, name=f'{name}.conv1')
    out = BatchNormalization(name=f'{name}.bn1')(out)
    out = Activation('relu', name=f'{name}.relu1')(out)

    out = conv3x3_bnn(out, planes, name=f'{name}.conv2')
    out = BatchNormalization(name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = Add(name=f'{name}.add')([identity, out])
    out = Activation('relu', name=f'{name}.relu2')(out)

    return out


def make_layer_bnn(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [tfp.layers.Convolution2DReparameterization(filters=planes,
                                                                kernel_size=1,
                                                                strides=stride,
                                                                kernel_divergence_fn = kl_divergence_function,
                                                                name=f'{name}.0.downsample.0'),
                      BatchNormalization(name=f'{name}.0.downsample.1')
                      ]

    x = basic_block_bnn(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block_bnn(x, planes, name=f'{name}.{i}')

    return x


def resnet_bnn(x, blocks_per_layer, num_classes=NUM_CLASSES):
    x = ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = tfp.layers.Convolution2DReparameterization(filters=64,
                                                    kernel_size=7,
                                                    strides=2, name='conv1',
                                                    kernel_divergence_fn = kl_divergence_function)(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer_bnn(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer_bnn(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer_bnn(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer_bnn(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D(name='avgpool')(x)
    # Create hidden layers with weight uncertainty using the DenseFlipout layer
    # Produces the parameters of the distribution
    distribution_params = tfp.layers.DenseReparameterization(units=num_classes,
                                                              activation="relu",
                                                              kernel_divergence_fn=kl_divergence_function,
                                                              name='fc')(x)
    # Output is a distribution object (OneHotCategorical), used for multiple classes
    outputs = tfp.layers.OneHotCategorical(num_classes)(distribution_params)

    return outputs

def resnet18_bnn(x, **kwargs):
    return resnet_bnn(x, [2, 2, 2, 2], **kwargs)