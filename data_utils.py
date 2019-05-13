import pickle
import keras
import time
import csv
import numpy as np
import matplotlib.pyplot as plt


class LossLog(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_begin(self,epoch,logs={}):
        self.losses_epoch = []

    def on_epoch_end(self,epoch,logs={}):
        self.losses.append(self.losses_epoch)

    def on_batch_end(self, batch, logs={}):
        self.losses_epoch.append(logs.get('loss'))

    def on_train_end(self,logs={}):
        layers=len(self.model.layers)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_loss_to_csv(self.losses,layers,timestr)

def save_pickle(path, obj):
    print('Saving object in: ' + path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    print('Loading object from: ' + path)
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_model(model,filename):
    model.save("Models/"+filename+".h5")    

def display_images(original,reconstructed):
    res=np.concatenate((original,reconstructed),axis=1)
    plt.imshow(res)
    plt.show()

def save_loss_to_csv(losses,layers,timestr):
    filepath="Logs/"+timestr+"_layers-"+str(layers)+".csv"

    with open(filepath, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for epoch in losses:
            writer.writerow(epoch)

    csvFile.close()
