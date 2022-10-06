from sklearn.model_selection import  train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
import os


weights_path = 'weights_iteration_{}'.format(1)# To create unique weight files, but can be made more efficient and better
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/ANN_weights2.hdf5'

train_dataset=pd.read_csv('D:/28_GitHub/Land Cover Mapping/MachineLearning/TrainDataset/PolSARData.csv')
X=train_dataset.iloc[:,:-1]
y=train_dataset.iloc[:,-1]
X=np.array(X)
y=np.array(y)
print(X)
print(y)
test_dataset1=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/Del78.csv")
test_dataset2=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/Del256.csv")
test_dataset3=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/SF116.csv")
test_dataset4=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/SF307.csv")

Test1=test_dataset1.iloc[:,:]
Test2=test_dataset2.iloc[:,:]
Test3=test_dataset3.iloc[:,:]
Test4=test_dataset4.iloc[:,:]

Test1=np.array(Test1)
Test2= np.array(Test2)
Test3=np.array(Test3)
Test4=np.array(Test4)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)
inp_shape=X_train.shape
print(inp_shape)
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=256, activation='linear',input_shape=9))
ann.add(tf.keras.layers.Dense(units=128, activation='selu'))
ann.add(tf.keras.layers.Dense(units=64, activation='selu'))
ann.add(tf.keras.layers.Dense(units=32,activation='linear'))
ann.add(tf.keras.layers.Dense(units=16,activation='selu'))
ann.add(tf.keras.layers.Dense(units=8,activation='linear'))
ann.add(tf.keras.layers.Dense(units=5, activation='sigmoid'))
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.1, patience=550, verbose=1, mode='auto')
model_checkpoint = ModelCheckpoint(weights_path, monitor='loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=10, min_lr=0.001)
ann.compile(optimizer=SGD(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ann.build(inp_shape)
ann.summary()
ann.fit(X_train,y_train,batch_size=256,epochs=700,validation_data=(X_test,y_test),callbacks=[model_checkpoint,early_stopping,reduce_lr])
ann.save("SavedModels/ANNModel2.h5")
y_pred1=ann.predict(Test1)
y_pred1=(y_pred1>0.3)
y_pred2=ann.predict(Test2)
y_pred2=(y_pred2>0.3)
y_pred3=ann.predict(Test3)
y_pred3=(y_pred3>0.3)
y_pred4=ann.predict(Test4)
y_pred4=(y_pred4>0.3)

z1=pd.DataFrame(y_pred1)
z2=pd.DataFrame(y_pred2)
z3=pd.DataFrame(y_pred3)
z4=pd.DataFrame(y_pred4)
z1.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/Del78_ANNResults2.csv")
z2.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/Del256_ANNResults2.csv")
z3.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/SF116_ANNResults2.csv")
z4.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/SF307_ANNResults2.csv")
#y_pred=(y_pred>0.5)
