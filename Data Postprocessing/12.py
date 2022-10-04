import tifffile as tiff
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import os
import cv2
import pandas as pd

def graycode(arr):
    gray_image=cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
    return gray_image 





print("Enter path to Ground Truth")#Ground truth annotation images
path1=input()
ground_truth_path=os.path.abspath(path1)
print("Enter path to Predicted Image")#Predicition from the models
path2=input()
predicted_image_path=os.path.abspath(path2)
ground_truth_image=tiff.imread(ground_truth_path)
predicted_image=tiff.imread(predicted_image_path)
#ground_truth_image=graycode(ground_truth_image)
#predicted_image=graycode(predicted_image)
truth=ground_truth_image.flatten()
prediction=predicted_image.flatten()
print(truth)
print(prediction)
confusionmatrix=confusion_matrix(truth,prediction,labels=[255,110,54,145,24])#[24,54,110,145,255]) Enter the labels corresponding to classes from 11.py
cmd=ConfusionMatrixDisplay(confusionmatrix)#,display_labels=['24','54','110','112','115','154','255']) To plot the actual confusion matrix
print(confusionmatrix)
cmd.plot()
np.savetxt('SFPap2.csv',confusionmatrix,delimiter=',')
'''

img1=pd.read_csv('Predict/GT.csv',delimiter=',')
img2=pd.read_csv('Predict/Pred.csv',delimiter=',')
img1=img1.to_numpy().flatten()
img2=img2.to_numpy().flatten()
con=confusion_matrix(img1,img2)
con
index=['Wetlands','Settlements','Water','Saltpan','Mangrove','Forest','Open Land']
col=['Wetlands','Settlements','Water','Saltpan','Mangrove','Forest','Open Land']
df=pd.DataFrame(con,index,col)
df
np.savetxt('Prediction.csv',img1,delimiter=',')
img1'''