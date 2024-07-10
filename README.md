# GAN-Generated Deepfake Image and AI-Generated Audio Detector

## Objective

With the increasing prevalence of AI-generated media, the need for robust detection mechanisms is paramount. This project aims to tackle the challenges of identifying GAN-generated deepfake images and AI-generated audio clips. By developing custom-built Convolutional Neural Networks (CNNs) and deploying them on Streamlit, this project offers an accessible and efficient solution for detecting deepfakes.

## Projects Overview

### 1. GAN-Generated Deepfake Image Detector

#### Architecture and Methodology
- **Preprocessing**: To enhance the model's ability to learn intrinsic features, the image dataset was preprocessed with Gaussian noise and blur.
- **Model**: A custom-built CNN with the following layers:
  - Convolutional layers with ReLU activation
  - Batch Normalization
  - Dropout layers for regularization
- **Accuracy**: The model achieved an accuracy of 94%.

### 2. AI-Generated Audio Detector

#### Architecture and Methodology
- **Preprocessing**: 2-second audio clips were transformed into respective spectrograms to capture time-frequency representation.
- **Model**: Another custom-built CNN with a similar architecture to the image detector, including:
  - Convolutional layers with ReLU activation
  - Batch Normalization
  - Dropout layers for regularization
- **Classification**: The model classifies audio clips into real or fake categories.

## Results
Both models have shown high accuracy in their respective tasks, demonstrating the effectiveness of the preprocessing techniques and the custom CNN architectures in detecting AI-generated media.

## Datasets Used
- **Image Dataset**: A collection of real and GAN-generated images, preprocessed with Gaussian noise and blur.
- **Audio Dataset**: 2-second clips of real and AI-generated audio, transformed into spectrograms.

## Contributors
- [Shreyans Garg](https://github.com/ShreyansGarg)
- [Contributor Name](https://github.com/contributor-github-username)

## Deployment
Both detectors are deployed on Streamlit, providing an easy-to-use web interface for real-time detection.
