from modules import *

#----------------------------------------------------------------------
# CONSTANT TO CHANGE
# If you want to change the block size, change this variable
# If you don't want any form of BTC, make the block_size = 1
# Make sure it's a power of 2 in this list: [1, 2, 4, 8, 16]
block_size = 16
#----------------------------------------------------------------------

BTC_directories([block_size])

Input,Output=load_data('../data/train_color', f'../data_btc_{block_size}')

print('Input shape is ' , Input.shape)
print('Output shape is ' , Output.shape)

Input=np.expand_dims(Input,axis=-1)
print('Input shape is ' , Input.shape)
print('Output shape is ' , Output.shape)

X_train, X_test, y_train, y_test = train_test_split(Input, Output, test_size=0.2, random_state=314, shuffle=False)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)

autoencoder = build_colorization_model()
autoencoder.summary()

checkpoint_cb =ModelCheckpoint(f"autoencoder_btc_{block_size}.h5",save_best_only=True)
autoencoder.compile(optimizer ='adam', loss='mse', metrics=['accuracy'])
hist=autoencoder.fit(X_train,y_train,epochs=100,validation_split=.1,callbacks=[checkpoint_cb])

print(autoencoder.evaluate(X_test,y_test))
predictions = autoencoder.predict(X_test)
predictions.shape

n = 10
plt.figure(figsize=(35, 25))
for i in range(n):
    # Display original gray images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i],cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display original color images
    ax = plt.subplot(3, n, i + 1)
    image = np.zeros((256, 256, 3))
    image[:, :, 0] = X_test[i][:,:,0]
    image[:, :, 1:] = y_test[i]* 128
    plt.imshow(lab2rgb(image))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display pred images
    ax = plt.subplot(3, n, i + 1 + n)
    image[:, :, 0] = X_test[i][:,:,0]
    image[:, :, 1:] = predictions[i]* 128
    plt.imshow(lab2rgb(image))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(f'../results/result_btc_{block_size}.png')
print(f'All results saved in "../results/result_btc_{block_size}.png"')