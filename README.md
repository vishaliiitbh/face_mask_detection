# Face Mask Detection Project

This project aims to detect whether a person is wearing a face mask in real-time using a webcam. The model is built using TensorFlow and Keras, 
and it leverages a Convolutional Neural Network (CNN) trained on a dataset of masked and unmasked faces.

## Project Overview

Face mask detection is a crucial task, especially during pandemics, to ensure that people are following safety protocols. 
This project uses deep learning techniques to detect whether a person is wearing a mask by processing live video feeds from a webcam.

## Dataset

The dataset used for training includes images of people with and without masks. The dataset can be obtained from Kaggle or any similar platform.

**Directory Structure:**
dataset/
│
├── train/
│ ├── mask/
│ └── no_mask/
├── val/
│ ├── mask/
│ └── no_mask/
└── test/
├── mask/
└── no_mask/

## Results
After running the real-time detection, the application will display a window with the live video feed from your webcam.
Detected faces will be marked with a rectangle and labeled with the prediction ("Mask" or "No Mask").
