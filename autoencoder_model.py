#%%

import keras
import numpy as np
import data_utils
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
from keras.models import load_model

x_train=data_utils.load_pickle("Data/x_train.pkl")
y_train=data_utils.load_pickle("Data/y_train.pkl")
x_val=data_utils.load_pickle("Data/x_val.pkl")
y_val=data_utils.load_pickle("Data/y_val.pkl")

image_size = 128 

#Randomly permutate the train/val sets:
p_train = np.random.permutation(len(x_train))
p_val = np.random.permutation(len(x_val))
x_train=x_train[p_train]
y_train=y_train[p_train]
x_val=x_val[p_val]
y_val=y_val[p_val]
#Smaller sets for fast training
x_train_small=x_train[p_train[1:100]]
y_train_small=y_train[p_train[1:100]]
x_val_small=x_val[p_val[1:100]]
y_val_small=y_val[p_val[1:100]]

#%%

##############
#Autoencoder:#
##############

#Source: https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/

def create_model():
    autoencoder = Sequential()

    # # Encoder Layers
    # autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3))) # 16 kernels of size 3x3 - The output is of size 16*size_image*size_image.
    # autoencoder.add(MaxPooling2D((2, 2), padding='same'))  #Reduction in size x2.
    # autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))
    # #autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # #autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    # # Flatten encoding for visualization
    # autoencoder.add(Flatten())
    # autoencoder.add(Reshape((16, 16, 8)))

    # # Decoder Layers
    # #autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # #autoencoder.add(UpSampling2D((2, 2)))
    # autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    # autoencoder.add(UpSampling2D((2, 2)))
    # autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # autoencoder.add(UpSampling2D((2, 2)))
    # autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same')) # 16 kernels of size 3x3 - The output is of size 16*size_image*size_image.
    # autoencoder.add(UpSampling2D((2, 2)))
    # autoencoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    # autoencoder.summary()

    autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3))) # 16 kernels of size 3x3 - The output is of size 16*size_image*size_image.
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))  #Reduction in size x2.
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))

    autoencoder.add(Conv2DTranspose(32, (5, 5), strides=(2,2), activation='relu', padding='same'))
    autoencoder.add(Conv2DTranspose(16, (3, 3), strides=(2,2), activation='relu', padding='same'))
    autoencoder.add(Conv2DTranspose(8, (2, 2), strides=(2,2), activation='relu', padding='same'))
    autoencoder.add(Conv2DTranspose(3, (3, 3), activation='relu', padding='same',))

    autoencoder.summary()

    #We build the model of the encoder
    # encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)

    return autoencoder


def fit_model(autoencoder,loss='mean_squared_error', optimizer = 'RMSprop',epochs=25,batch_size=128,save=True):
    history = data_utils.LossLog()

    #autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(loss=loss,optimizer=optimizer)

    if(save):
        autoencoder.fit(x_train, x_train, batch_size, epochs,validation_data=(x_val, x_val),callbacks=[history])
        layer_nr=len(autoencoder.layers)
        model_filename = str(layer_nr)+'l_'+str(epochs)+'e_'+str(batch_size)+'bsize-'+loss+'-'+optimizer
        data_utils.save_model(autoencoder,model_filename)
    else:
        autoencoder.fit(x_train, x_train, batch_size, epochs,validation_data=(x_val, x_val))
        
    return autoencoder

#Bigger batch size -> Bigger generalization

#%%

autoencoder=create_model()
autoencoder=fit_model(autoencoder,epochs=25,optimizer = 'adam',batch_size=32)

#%%
img_x = x_val[1:1050]
img = autoencoder.predict(img_x)

data_utils.display_images(img_x[4],img[4])

#%% Save the model


#To load the model:
#autoencoder = load_model(path)



