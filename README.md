# Skin Lesion Segmentation

This repository contains a complete implementation of a U-Net model for skin lesion image segmentation. The U-Net model is a widely used architecture for various image segmentation tasks, including medical image analysis.

## Description

This implementation is based on the provided Jupyter Notebook (`main.ipynb`). The notebook code includes the following key components:

1. **Data Loading and Preprocessing:** The code starts by loading metadata and images from the dataset. The dataset is preprocessed to ensure uniform dimensions and pixel values suitable for training.

2. **U-Net Model Architecture:** A U-Net model architecture is defined using TensorFlow and Keras. This architecture is specifically designed for image segmentation tasks. It consists of an encoder-decoder structure with skip connections.

3. **Model Training:** The code compiles the U-Net model using the Adam optimizer and binary cross-entropy loss function. It also incorporates data augmentation techniques to enhance the model's ability to generalize. The training process is visualized and monitored, and early stopping is implemented to prevent overfitting.

4. **Model Evaluation:** After training, the model is evaluated on a test set. Various metrics, including loss, accuracy, confusion matrix, classification report, ROC curve, and AUC score, are computed to assess the model's performance.

5. **Visualization:** The code randomly selects a sample from the test set and displays the original image, ground truth mask (segmentation), and predicted mask. This visualization helps users interpret the model's segmentation results.

## Usage

To run the code and experiment with the U-Net model for skin lesion segmentation, follow these steps:

1. **Install Dependencies:** Install the required Python packages by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. **Dataset Download:** Download the skin lesion dataset from the ISIC Challenge website using the following link: [Dataset Download](https://challenge.isic-archive.com/data/). Ensure that the dataset is organized appropriately and that image files and corresponding masks are available.

3. **Data Augmentation (Optional):** If desired, adjust data augmentation parameters in the code to enhance model training. Data augmentation helps improve the model's robustness.

4. **Training:** Open and execute the provided Jupyter Notebook (`main.ipynb`). The notebook guides you through model training, evaluation, and visualization. Follow the instructions within the notebook to train and assess the model's performance.

Feel free to modify the code and parameters to adapt it to your specific skin lesion segmentation project. The provided Jupyter Notebook serves as a comprehensive starting point for developing your image segmentation models.

## Notebook Code

You can find the complete implementation in the [main.ipynb](main.ipynb) notebook. This notebook includes the code for data loading, model architecture, training, evaluation, and visualization. It's designed to be self-contained and easy to follow.

Please note that you should have TensorFlow, Keras, and other required libraries installed to run the notebook effectively.

This repository provides a structured and comprehensive approach to skin lesion segmentation using a U-Net model.
