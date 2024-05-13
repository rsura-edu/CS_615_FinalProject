from PIL import Image
import os
import numpy as np

# Block Truncation Coding for Gray Scale Images
def BTC(img, block_size):
    if isinstance(img, str):
        # Load the image if img is a path
        img = Image.open(img)
        img = np.array(img)
    elif isinstance(img, Image.Image):
        img = np.array(img)
    elif not isinstance(img, np.ndarray):
        raise TypeError("Input must be a NumPy array or a path to an image file.")
    
    try:
        block_size = abs(int(block_size))
    except:
        block_size = 1
    
    if block_size < 2:
        return img
    
    rows, cols = 0,0
    if len(img.shape) == 3:
        rows, cols = img.shape[:2]
        img = img[:, :, 0]
    else:
        rows, cols = img.shape
    I_new = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(rows // block_size):
        for j in range(cols // block_size):
            # get block
            row_start = block_size * i
            col_start = block_size * j
            subgrid = img[row_start:row_start + block_size, col_start:col_start + block_size]

            subgrid = subgrid.astype(float)  # so that subtraction isn't bounded

            avg_intensity = np.sum(subgrid) / (block_size ** 2)
            
            standard_dev = 0
            for u in range(block_size):
                for v in range(block_size):
                    standard_dev += (subgrid[u, v] - avg_intensity) ** 2
                    
            standard_dev = np.sqrt(standard_dev / (block_size ** 2))

            binary_block = subgrid >= avg_intensity
            # Decode
            Q = np.sum(binary_block)
            P = (block_size ** 2) - Q
            if P != 0:
                A = np.sqrt(Q / P)
                if A != 0:  # Add check to avoid division by zero
                    subgrid[binary_block == True] = avg_intensity + (standard_dev / A)
                    subgrid[binary_block == False] = avg_intensity - (standard_dev * A)
                else:
                    subgrid[binary_block == True] = avg_intensity
                    subgrid[binary_block == False] = avg_intensity


            # Append to output image
            I_new[row_start:row_start + block_size, col_start:col_start + block_size] = subgrid.astype(np.uint8)
    
    # # Duplicate the gray channel if it's single-channel to produce an RGB image
    if len(img.shape) == 2:
        I_new = np.stack([I_new] * 3, axis=-1)
    
    # # Make a 3-D, 1-channel to make a grayscale image that's shape size 3
    # if len(img.shape) == 2:
    #     I_new = np.expand_dims(img,axis=-1)
    
    return I_new

def BTC_directories(blocks=[1, 2, 4, 8, 16]):
    # Define the paths
    data_folder = "../data/train_black"
    for block in blocks:
        counter = 0
        output_folder = f"../data_btc_{block}"
        if os.path.exists(output_folder):
            print(f"Output folder for block size {block} already exists. Skipping data BTC preprocessing...")
            continue
        # Iterate through the subfolders
        for root, dirs, files in os.walk(data_folder):
            # create the corresponding subfolder in the output directory
            output_subfolder = os.path.join(output_folder, os.path.relpath(root, data_folder))
            os.makedirs(output_subfolder, exist_ok=True)

            # Iterate through the files in the current subfolder
            for file in files:
                # Check if the file is a jpg file
                if file.lower().endswith(".jpg"):
                    # making the input and output file paths
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_subfolder, file.replace(".jpg", ".png"))

                    # apply BTC function and save the converted image
                    Image.fromarray(BTC(input_file, block)).convert('RGB').save(output_file)
                    
                    # progress indicator
                    counter += 1
                    if counter % 50 == 0:
                        print(f"Processed {counter} images with block size {block}")