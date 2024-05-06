from PIL import Image
import os
import numpy as np
import cv2

def resize_image(image_path, output_size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.resize(output_size, Image.ANTIALIAS)
        return img
    

# Load and append all test and train images
def load_images(data_dir):
    images = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = Image.open(img_path)
        images.append(img)
    return images

# All Gray Scale Imgs are in ../data/train_black and ../data/test_black
train_gray_scale_images = load_images('../data/train_black')
test_gray_scale_images = load_images('../data/test_black')

# All RGB Imgs are in ../data/train_color and ../data/test_color
train_rgb_images = load_images('../data/train_color')
test_rgb_images = load_images('../data/test_color')


# Histogram Equalization for Gray Scale Images
def histogram_equalization_3channel(img):
    if isinstance(img, str):
        # Load the image if img is a path
        img = cv2.imread(img)
        # Convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif not isinstance(img, np.ndarray):
        raise TypeError("Input must be a NumPy array or a path to an image file.")

    # Check if the image is indeed in a three-channel format
    if img.shape[2] != 3:
        raise ValueError("Image does not have three channels.")
    
    # Since the image is grayscale but in three channels, just take one channel for processing
    gray_channel = img[:, :, 0]
    
    # Apply histogram equalization
    equalized_channel = cv2.equalizeHist(gray_channel)
    
    # Replicate the equalized channel across all three channels
    equalized_image = np.stack([equalized_channel] * 3, axis=-1)
    
    # Convert the image back to BGR from RGB
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR)
    
    return equalized_image

# Block Truncation Coding for Gray Scale Images

def BTC(img, block_size):
    if isinstance(img, str):
        # Load the image if img is a path
        img = cv2.imread(img, 0)
    elif not isinstance(img, np.ndarray):
        raise TypeError("Input must be a NumPy array or a path to an image file.")
    
    rows, cols = img.shape
    I_new = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(rows // block_size):
        for j in range(cols // block_size):
            # get block
            row_start = block_size * i
            if row_start >= rows:
                break
            col_start = block_size * j
            if col_start >= cols:
                break
            subgrid = img[row_start:min(row_start + block_size, rows), col_start:min(col_start + block_size, cols)]

            subgrid = subgrid.astype(float)  # so that subtraction isn't bounded

            avg_intensity = np.sum(subgrid) / (block_size ** 2)
            standard_dev = np.sqrt(np.sum((subgrid - avg_intensity) ** 2) / (block_size ** 2))

            binary_block = np.uint8(subgrid >= avg_intensity)

            # Decode
            Q = np.sum(binary_block)
            P = (block_size ** 2) - Q
            if P != 0:
                A = np.sqrt(Q / P)
                subgrid[binary_block == 1] = avg_intensity + (standard_dev / A)
                subgrid[binary_block == 0] = avg_intensity - (standard_dev * A)
            else:
                subgrid[binary_block == 1] = avg_intensity
                subgrid[binary_block == 0] = avg_intensity
            if len(set(list(subgrid.reshape(block_size ** 2)))) > 2:
                print(subgrid)
            # Append to output image
            I_new[row_start:min(row_start + block_size, rows), col_start:min(col_start + block_size, cols)] = subgrid.astype(np.uint8)
    
    # Duplicate the gray channel 3 times to produce an RGB image
    I_new = np.stack([I_new] * 3, axis=-1)
    
    return I_new
