import numpy as np
import tifffile as tiff

for i in range(1,5):#Enter the total images obtained after horizontal stitch (8.py)
    img1=tiff.imread('D:/1_Dataset/GeoSpatial/Data/Mumbai/Tiff/T3/Tiff/Mumbai/PauliRGB/Pred{}.tif'.format(i))#Enter the name of the directory
    if i==1:
        
        d=img1
    else:

    
        c=np.concatenate((d,img1),axis=0)
        d=c
        

tiff.imsave('D:/1_Dataset/GeoSpatial/Data/Mumbai/Tiff/T3/Tiff/Mumbai/PauliRGB/Prediction.tif',d)#Enter the save directory for the recombined feature map

'''
img1=tiff.imread('C:/Users/rrkot/OneDrive/Desktop/Semantic Results/Mumbai/trial/Prediction.tif')
img1=img1.transpose([1,2,0])
tiff.imsave('C:/Users/rrkot/OneDrive/Desktop/Semantic Results/Mumbai/trial/4.tif',img1)
'''