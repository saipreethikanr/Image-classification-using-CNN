# 🧠 Image Classification with Convolutional Neural Networks (CNNs) on CIFAR-10

## 🚀 Project Overview
This project showcases an end-to-end image classification pipeline using **Convolutional Neural Networks (CNNs)** on the **CIFAR-10** dataset. It compares CNNs with traditional **Artificial Neural Networks (ANNs)** to highlight the superior performance of CNNs in handling complex image data.

The goal is to build, train, and evaluate deep learning models to accurately classify images into one of 10 categories.

---

## ✨ Features

- **CIFAR-10 Dataset Handling**  
  Efficient loading, exploration, and visualization of 60,000 32x32 RGB images across 10 classes.

- **Data Preprocessing**  
  Normalization of image pixel values to the [0, 1] range for faster and more stable training.

- **ANN Baseline Model**  
  Implementation of a simple ANN to establish a baseline, demonstrating its limitations on image data.

- **CNN Model Architecture**  
  Construction of a robust CNN architecture using Conv2D, MaxPooling, Flatten, and Dense layers.

- **Model Training & Evaluation**  
  Training the CNN model and evaluating its performance using accuracy and classification reports (precision, recall, F1-score).

- **Prediction Examples**  
  Predicting class labels for unseen test images using the trained model.

---

## 🛠️ Technologies Used

- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow & Keras  
- **Numerical Computing**: NumPy  
- **Data Visualization**: Matplotlib  
- **Model Type**: Convolutional Neural Networks (CNNs)

---

## 📊 Dataset: CIFAR-10

The CIFAR-10 dataset is a standard benchmark in image classification tasks and contains:

- 60,000 images (32x32 pixels, RGB)
- 10 classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck
- 50,000 training images  
- 10,000 testing images

---

## 🧠 Model Architecture (CNN)

Typical CNN architecture used in this project includes:

- **Convolutional Layers** (`Conv2D`)  
  Extract spatial features using multiple filters.

- **Activation Functions** (`ReLU`)  
  Introduce non-linearity.

- **Pooling Layers** (`MaxPooling2D`)  
  Downsample the feature maps to reduce complexity.

- **Flatten Layer**  
  Convert 2D feature maps into a 1D vector.

- **Dense Layers**  
  Fully connected layers for classification.

- **Output Layer**  
  `Softmax` activation for multiclass probability output.

---

## 📈 Results

- **ANN Test Accuracy**: ~47-48%  
- **CNN Test Accuracy**: ~70%

✅ The CNN model clearly outperforms the ANN baseline, emphasizing its ability to automatically learn spatial hierarchies in image data.

---

