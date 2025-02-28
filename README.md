# Arrhythmia-on-ECG-Classification-using-CNN
##Overview

This project implements an Electrocardiogram (ECG) signal classification model using a 1D Convolutional Neural Network (CNN). The goal is to classify ECG signals into five categories based on the MIT-BIH Arrhythmia Dataset.

##Dataset

The dataset used for training and testing is the MIT-BIH Arrhythmia Database, obtained from PhysioNet. It contains labeled ECG signals categorized into different types of heartbeats.

##Preprocessing

Before training the model, the raw ECG signals undergo the following preprocessing steps:

Normalization: Standardizing the signal values for uniformity.

Resampling: Adjusting the sampling rate to a consistent frequency.

Segmentation: Extracting fixed-length segments from the ECG signals.

Label Encoding: Converting categorical labels into numerical values.

One-Hot Encoding: Transforming labels into a one-hot representation for classification.

##Model Architecture

The model is built using Keras and TensorFlow and consists of the following layers:

Input Layer: Takes 1D ECG signal segments as input.

Convolutional Layers: Three 1D convolutional layers with ReLU activation.

Batch Normalization: Applied after each convolutional layer for faster training.

MaxPooling Layers: Downsamples the feature maps to reduce dimensionality.

Flatten Layer: Converts feature maps into a single vector.

Fully Connected Layers: Two dense layers with ReLU activation.

Output Layer: A softmax layer with five output classes.

##Training

The model is trained using:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 32

Epochs: 40 (with early stopping)

Validation Split: 20% of the dataset

##Results

The model achieved an accuracy of 76.78% on the test set. Below are key observations:

Training Accuracy steadily increased, showing good learning capability.

Validation Accuracy fluctuated, indicating possible overfitting.

Confusion Matrix Analysis:

Normal and Q classes were classified well.

Some misclassifications occurred between S, V, and F classes.

##Performance Evaluation

Accuracy & Loss Curves

The training accuracy increased over epochs, while validation accuracy fluctuated.

Training loss decreased consistently, but validation loss showed instability.

##Confusion Matrix Analysis

The model performed well for normal beats but struggled to differentiate between abnormal classes.

Further improvements can be made using additional feature engineering and hyperparameter tuning.
