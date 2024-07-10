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
- **Accuracy**: The model achieved an accuracy of 85%.

## Results
Both models have shown high accuracy in their respective tasks, demonstrating the effectiveness of the preprocessing techniques and the custom CNN architectures in detecting AI-generated media.

## Datasets Used
- **Image Dataset**: The dataset which we used for this project is “140k Real and Fake Faces” from Kaggle [[Dataset Link](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)]
- **Audio Dataset**: 2-second clips of real and AI-generated audio, transformed into spectrograms.

## Contributors
- [Shreyans Garg](https://github.com/ShreyansGarg)
- [Mahua Singh](https://github.com/S-Mahua)
- [Vishal Chaudhary](https://github.com/cvishal-19)
- [Malabh Bakshi](https://github.com/Malabh)

## Deployment
Both detectors are deployed on Streamlit, providing an easy-to-use web interface for real-time detection.
- [Image](https://deepfake-detection-using-cnns.streamlit.app/)
- [Audio](https://deepfake-audio-detection.streamlit.app/)
