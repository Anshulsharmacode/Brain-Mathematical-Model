# Theory for Lung Tumor Prediction Application

## Overview

The Lung Tumor Prediction application consists of two primary components: a FastAPI backend that processes images and predicts tumor presence, and a Streamlit frontend that allows users to upload images and view predictions. This document provides a theoretical background of the concepts used in the application.

## Key Concepts

### 1. Image Processing

#### Grayscale Conversion
To simplify the analysis, images are often converted to grayscale. This is done by calculating a single intensity value for each pixel, which is a weighted combination of the red, green, and blue color components. Grayscale images reduce the complexity of the data and make it easier to extract relevant features.

### 2. Feature Extraction

#### Intensity Statistics
Statistical measures such as mean intensity, standard deviation, and coefficient of variation are extracted from the image. These measures help in understanding the distribution of intensity values in the image, which is crucial for distinguishing between tumor and non-tumor regions.

#### Edge Detection
Edge detection is a critical step in identifying boundaries within images. The Sobel filter is commonly used to detect edges by highlighting areas of high spatial frequency. This process assists in locating regions that may indicate the presence of a tumor.

### 3. Texture Features using Gray Level Co-occurrence Matrix (GLCM)
The GLCM is a statistical method for examining texture that considers the spatial relationship of pixels. Various texture features can be derived from the GLCM, which provide insights into the texture patterns of the image. These features are important in the classification of tumors based on their texture characteristics.

### 4. Model Prediction
The prediction of tumor presence is performed using a Random Forest Classifier. This machine learning model learns from the extracted features during the training phase, enabling it to classify new images as either containing a tumor or being non-tumorous. The model aggregates the predictions from multiple decision trees to improve accuracy and reduce the risk of overfitting.

## Conclusion
This document outlines the theoretical foundations used in the Lung Tumor Prediction application. By converting images to grayscale, extracting statistical features, and utilizing the GLCM for texture analysis, the application aims to effectively classify MRI images as containing a tumor or not.
