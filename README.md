# HDR-ML-Challenge-Hackathon

# Overview
This guide provides instructions for running the backend of the submitted anomaly detection model. The model predicts anomalies based on input features like Sea Level Anomaly (SLA), latitude, and longitude. It uses a trained neural network saved as a checkpoint file (model.pth). This document outlines the necessary setup, data requirements, and execution steps.


# Summary Description of the Model
The anomaly detection model leverages a neural network. The network consists of fully connected layers with ReLU activations and dropout for regularization. It outputs the probability of an anomaly for a given set of inputs, which is thresholded to generate binary predictions (0 for normal and 1 for anomaly). The model was trained using data from satellites and stations. Training used a binary cross-entropy loss function optimized with the Adam optimizer. The model's architecture and trained parameters are stored in the model.pth file, ensuring consistent predictions during inference.

# Prerequisites
# Environment Setup
Ensure the following are installed:

Python 3.8 or later
Required libraries: pandas, numpy, torch, scikit-learn, and matplotlib
Install these dependencies using the provided requirements file.

# Input Files
The required input files include:

NetCDF files containing SLA data, with variables for SLA, latitude, and longitude.
A station metadata CSV file with details about station names, latitude, and longitude.
# Model Checkpoint
The file model.pth contains the trained weights and biases of the neural network and is used for making predictions.

# Execution Steps
Load the Model
The model architecture must match the training configuration to load the checkpoint correctly. Ensure the model is set to evaluation mode to avoid updating weights during prediction.

# Process Input Data
The backend reads and processes NetCDF files to extract SLA data for the required dates and locations. Station metadata is parsed to gather station coordinates, which are combined with SLA data for input to the model.

# Generate Predictions
For each date and station:

SLA data is extracted for the specific date and combined with the station’s latitude and longitude to form input features.
The model predicts the probability of an anomaly based on these features.
Probabilities are converted to binary predictions using a threshold (e.g., 0.5).
Save Results
Predictions are saved in a CSV file. The file is structured with dates as rows, station names as columns, and anomaly values (0 or 1) as entries.

# Save Results
Predictions are saved in a CSV file. The file is structured with dates as rows, station names as columns, and anomaly values (0 or 1) as entries.

