# VisionAI CIFAR-10 Classifier

A deep learning image classification project built with TensorFlow/Keras using Convolutional Neural Networks (CNN) trained on the CIFAR-10 dataset.

---

## Overview

This project classifies 32x32 color images into one of 10 distinct categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The trained model is exported as `image_classifier_cnn.keras` for downstream inference or web deployment.

---

## Repository Structure

```text
visionai-cifar10-classifier/
├── ai image/                  # Sample test images / test inputs
├── static/                    # Static assets (UI styles, scripts, or images)
├── .gitignore                 # Files and folders ignored by Git
├── image_classifier_cnn.keras # Pre-trained Keras CNN model weights
└── README.md                  # Project documentation
