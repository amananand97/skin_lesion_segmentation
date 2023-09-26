import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress updates
from IPython.display import clear_output  # For clearing terminal output

# Import the cv2 library
import cv2

# Define a function to save the mask images
def save_mask(path, mask):
    """Saves the mask image to the specified path.

    Args:
        path (str): The path of the image.
        mask (numpy.ndarray): The mask image.
    """
    mask_name = os.path.basename(path)[:-4] + '_mask.jpg'
    mask_path = os.path.join('masks', mask_name)
    cv2.imwrite(mask_path, mask)

# Create the masks directory if it doesn't exist
if not os.path.exists('masks'):
    os.mkdir('masks')

# Load metadata
metadata_path = 'data/metadata.csv'
metadata = pd.read_csv(metadata_path)

# Load and preprocess images
image_paths = ['data/' + img_id + '.jpg' for img_id in metadata['isic_id']]

# Define target size for resizing images
target_size = (256, 256)

# Convert to grayscale and apply thresholding
masks = []
for path in tqdm(image_paths, desc='Generating Masks'):
    img = tf.image.decode_jpeg(tf.io.read_file(path))
    img = tf.image.resize(img, target_size)
    img = img / 255.0

    # Convert TensorFlow tensor to NumPy array
    img_np = img.numpy()

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Convert to 8-bit integer image
    gray_8bit = cv2.convertScaleAbs(gray)

    # Apply thresholding to generate the mask
    _, mask = cv2.threshold(gray_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the mask image
    save_mask(path, mask)



# Visualize the generated masks
# plt.figure(figsize=(10, 4))
# for i in range(4):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(images[i])
#     plt.title('Image')
#     plt.axis('off')

#     plt.subplot(2, 4, i+5)
#     plt.imshow(masks[i], cmap='gray')
#     plt.title('Generated Mask')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()