# Comparative Analysis of CNN, ResNet, VGG, ANN, and InceptionNet for Predicting Vitiligo Progression using Dermatological Images

Welcome to the repository for our research project on predicting vitiligo progression using advanced deep learning models. This project focuses on comparing the performance of various architectures including Convolutional Neural Networks (CNN), ResNet, VGG, Artificial Neural Networks (ANN), and InceptionNet. 

## Project Overview

Vitiligo is a skin condition characterized by the loss of skin color in patches. Accurate prediction of vitiligo progression can aid in better management and treatment of the condition. This project explores the potential of various deep learning models in predicting vitiligo progression using dermatological images.

## Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project consists of dermatological images of vitiligo-affected and healthy skin. It is organized into training and validation sets.

## Preprocessing

Image preprocessing includes normalization, data augmentation, and resizing to 224x224 pixels. Data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping are applied to enhance the dataset.

## Model Architectures

### Custom CNN Model
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
