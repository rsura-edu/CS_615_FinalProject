import os 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.models import load_model
import keras 
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab,lab2rgb
from skimage.metrics import peak_signal_noise_ratio

from preprocess import *

def build_colorization_model(input_shape=(256,256,1)):
    input_ = keras.layers.Input(shape=input_shape)
    # Encoder
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',strides=2)(input_)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',strides=2)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',strides=2)(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    encoder = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    # Decoder
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(encoder)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    # Adjust the output layer for an RGB image (2 channels)
    x = keras.layers.Conv2D(2, (3, 3), padding='same', activation=keras.layers.LeakyReLU(alpha=.5))(x)
    decoder = keras.layers.UpSampling2D((2, 2))(x)
    # Autoencoder model
    model = keras.models.Model(inputs=input_, outputs=decoder)
    
    return model

# Modify the load_data function to return the correct shape for the input data
def load_data(color_path, btc_path, num_images=5000):
    Input, Output=[],[]
    for i in tqdm(sorted(os.listdir(color_path))):
        if not os.path.exists(os.path.join(btc_path,i)):
            btc_path_ = os.path.join(btc_path,i.replace('.jpg', '.png'))
        else:
            btc_path_ = os.path.join(btc_path,i)
        input_image = load_img(btc_path_,target_size=(256,256),color_mode='rgb')
        input_image=img_to_array(input_image)
        input_image=input_image/255.0
        lab=rgb2lab(input_image)
        Input.append(lab[:,:,0])
        
        color_path_ = os.path.join(color_path,i)
        output_image = load_img(color_path_,target_size=(256,256),color_mode='rgb')
        output_image = img_to_array(output_image)
        output_image = output_image/255.0
        lab = rgb2lab(output_image)
        Output.append(lab[:,:,1:]/128)
    return np.array(Input), np.array(Output)

def evaluate_model(model, X, y):
    eval_loss, eval_accuracy = model.evaluate(model.predict(X), y)
    print('Evaluation Loss:', eval_loss)
    print('Evaluation Accuracy:', eval_accuracy)

def visualize_results(X, y, predictions, n, block_size):
    if not os.path.exists(f'../results/'):
        os.makedirs(f'../results/')
    # plt.figure(figsize=(35, 25))
    plt.figure()
    extension_length = len(str(n))
    for i in range(n):
        # Display original gray images
        ax = plt.subplot(1, 3, 1)
        plt.imshow(X[i],cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Gray')
        
        # Display original color images
        ax = plt.subplot(1, 3, 2)
        image = np.zeros((256, 256, 3))
        image[:, :, 0] = X[i][:,:,0]
        image[:, :, 1:] = y[i]* 128
        plt.imshow(lab2rgb(image))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Color')
        
        # Display pred images (colorized form)
        ax = plt.subplot(1, 3, 3)
        image = np.zeros((256, 256, 3))
        image[:, :, 0] = X[i][:,:,0]
        image[:, :, 1:] = predictions[i]* 128
        plt.imshow(lab2rgb(image))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Predicted Color Image')
        
        # save into results folder
        formatted_number = ('{:0' + str(extension_length) + 'd}').format(i)
        if not os.path.exists(f'../results/btc_{block_size}'):
            os.makedirs(f'../results/btc_{block_size}')
        
        plt.savefig(f'../results/btc_{block_size}/result_{formatted_number}.png')
    
    print(f'All results saved in ../results/btc_{block_size}')

def train_model(block_size=1):
    Input, Output=load_data('../data/train_color', f'../data_btc_{block_size}', block_size)
    print("Input shape:", Input.shape)
    print("Output shape:", Output.shape)
    # Input=np.expand_dims(Input,axis=-1)
    # print("Input shape:", Input.shape)
    X_train, X_test, y_train, y_test = train_test_split(Input, Output, test_size=0.2, random_state=314, shuffle=False)
    
    autoencoder : object
    model_fp = f"autoencoder_btc_{block_size}.h5"
    try:
        autoencoder = load_model(model_fp)
        print(f'Model of BTC block size {block_size} already exists. Skipping training.')
    except (FileNotFoundError, OSError):
        autoencoder = build_colorization_model()

        # print(autoencoder.summary())

        checkpoint_cb = ModelCheckpoint(model_fp, save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
        autoencoder.compile(optimizer ='adam', loss='mse', metrics=['accuracy'])
        hist=autoencoder.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=.2, callbacks=[checkpoint_cb, early_stopping_cb])
    except:
        raise Exception('Model training failed')

    predictions = autoencoder.predict(X_train)
    
    # Input, Output=load_data('../data/train_color', f'../data_btc_1', 1)
    # print("Input shape:", Input.shape)
    # print("Output shape:", Output.shape)
    # X_train, _, _, _ = train_test_split(Input, Output, test_size=0.2, random_state=314, shuffle=False)
    
    print('Statistics on model trained on BTC with block size:', block_size)
    # evaluate_model(autoencoder, X_train, y_train)
    
    
    visualize = input('Visualize the model? (y/n)')
    visualize = True if 'y' in visualize.lower() else False
    
    if visualize:
        num_images = int(input('Enter the number of images to visualize: '))
        visualize_results(X_train, y_train, predictions, num_images, block_size)

