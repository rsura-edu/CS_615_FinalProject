from PIL import Image
import os

def resize_image(image_path, output_size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.resize(output_size, Image.ANTIALIAS)
        return img
    
# All Gray Scale Imgs are in ../data/train_black and ../data/test_black
# All RGB Imgs are in ../data/train_color and ../data/test_color
# Load and append all test and train gray scale images
def load_gray_scale_images(data_dir):
    gray_scale_images = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = Image.open(img_path)
        gray_scale_images.append(img)
    return gray_scale_images

# Load and append all test and train RGB images
def load_rgb_images(data_dir):
    rgb_images = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = Image.open(img_path)
        rgb_images.append(img)
    return rgb_images

train_gray_scale_images = load_gray_scale_images('../data/train_black')
test_gray_scale_images = load_gray_scale_images('../data/test_black')

train_rgb_images = load_rgb_images('../data/train_color')
test_rgb_images = load_rgb_images('../data/test_color')


# Histogram Equalization for Gray Scale Images


# Block Truncation Coding for Gray Scale Images