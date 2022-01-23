from Architecture import *
from data_augmentation import *

import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from  sklearn.metrics import confusion_matrix

import time
from tqdm.keras import TqdmCallback



def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    yz=img/(img.max())
    return x



N_BANDS = 9# T3 elements
N_CLASSES = 7  # Open Land, Water, Settlement, Forest, Mangrove, Saltpan, Wetland 
CLASS_WEIGHTS = [0.01, 0.1, 0.5, 0.6, 0.8, 0.8, 0.9]# Decreasing order of frequency of class
N_EPOCHS = 500# number of epochs
UPCONV = True
PATCH_SZ = 512   # patch size of the tiles
BATCH_SIZE = 4
TRAIN_SZ = 60  # train size for data augmentation. can be higher if GPU has enough VRAM
VAL_SZ =  30  # validation size for data augmentation, Can be higher if GPU has enough VRAM


def get_model():
    return Unet(N_CLASSES, im_sz=PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights_iteration_{}'.format(int(time.time()))# To create unique weight files, but can be made more efficient and better
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = [str(i).zfill(1) for i in range(1,79)]  # Please change based on available training images, start from 1 to n+1, where n is the total images inside the directory
validIds= [str(i).zfill(1) for i in range(1,18)]  # Please change based on available validaton images, start from 1 to n+1, where n is the total images inside the directory

if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading Training images')
    for img_id in trainIds:
        img_m = (tiff.imread('/content/drive/My Drive/Project/DataSet/Training_Images/{}.tif'.format(img_id))).transpose([1,2,0])#Please enter valid directory name and add the .transpose method as shown
        mask = tiff.imread('/content/drive/My Drive/Project/DataSet/y/{}.tif'.format(img_id)).transpose([1, 2, 0])/255 #Please enter valid directory name and add the .transpose method as shown
        X_DICT_TRAIN[img_id] = img_m[:, :, :]
        Y_DICT_TRAIN[img_id] = mask[:, :, :]
        print(img_id + '--> Training Image read')

    print('Reading Validation images')
    for img_id in validIds:
        img_m = (tiff.imread('/content/drive/My Drive/Project/DataSet/Training_Images/{}.tif'.format(img_id))).transpose([1,2,0])#Please enter valid directory name and add the .transpose method as shown
        mask = tiff.imread('/content/drive/My Drive/Project/DataSet/y/{}.tif'.format(img_id)).transpose([1, 2, 0])/255#Please enter valid directory name and add the .transpose method as shown
        X_DICT_VALIDATION[img_id] = img_m[:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[:, :, :]
        print('Validation Image:',img_id + '--> read')


    print('Images were read')

    def train_net():
        print("Start train network")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        NAME="UNet-PolSARImage_512x512_{}".format(int(time.time()))

        if os.path.isfile(weights_path):
            model.load_weights(weights_path)

         
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.1, patience=500, verbose=1, mode='auto')
        
        model_checkpoint = ModelCheckpoint(weights_path, monitor='loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        log_diri = './content/sample_data/{}'.format(NAME) #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")/
        tensorboard = TensorBoard(log_dir=log_diri, write_graph=True, write_images=True,update_freq='epoch')
        
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                   shuffle=True,#verbose=2,
                  callbacks=[model_checkpoint, csv_logger, tensorboard,early_stopping,TqdmCallback(verbose=0)],
                  validation_data=(x_val, y_val))
        
        return model

    train_net()