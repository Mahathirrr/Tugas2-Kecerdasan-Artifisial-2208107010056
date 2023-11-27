# Brain Tumor Detection with Deep Learning

This repository contains code for detecting brain tumors using MRI images with Convolutional Neural Networks (CNN).

## Authors

Muhammad Mahathir (2208107010056)

## Dataset
The dataset used is [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) available on Kaggle. This dataset consists of human brain MRI images and labels indicating whether the images show a tumor (1) or not (0).

## Convolutional Neural Networks (CNN)

CNN is a type of artificial neural network highly effective for image processing. CNN can automatically and adaptively learn important features from images through the training process. In this project, CNN is used to learn features from brain MRI images and classify whether the image indicates the presence of a tumor or not.

## Installation

1. Ensure you have Python installed.
2. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the brain MRI image dataset (brain_tumor_dataset) and place it inside the `dataset` directory.
2. Open and run the `brain_tumor_detector.ipynb` notebook or execute the `brain_tumor_detector.py` script to train and evaluate the model.

## Project Structure

- `data/`: Directory containing the brain MRI image dataset (brain_tumor_dataset).
- **requirements.txt:** File listing dependencies.
- **brain_tumor_detector.ipynb:** Jupyter notebook for training and evaluating the model.
- **brain_tumor_detector.py:** Python script for the same purposes.
- `laporan/`: Directory containing report files.  
  - **Spam Email Detector.pptx:** PowerPoint presentation with slides about the repository and model.
- `models/`: Directory containing saved models.
  - **Spam-Detector.h5:** File representing the saved model.
- `screenshot/`: Directory containing screenshots accuracy and plot model.
