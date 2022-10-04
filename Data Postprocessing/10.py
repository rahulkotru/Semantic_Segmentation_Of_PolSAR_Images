import numpy as np
import cv2
import tifffile as tiff
import os

def remove_ext(stringwe):
    base=os.path.basename(stringwe)
    name=os.path.splitext(base)[0]
    return name

def save_img(arr,path,name):
    print("Saving Images...\n"+name)
    
    tiff.imsave(path+'/{}.tif'.format(name),arr)

def slice(num,img):
    (x,y)=img.shape
    flat_img=img.flatten()
    for j in range(len(flat_img)):
        if flat_img[j]==num:
            flat_img[j]=255
        else:
            flat_img[j]=0
    return flat_img.reshape(x,y)


def grayslice(file,path): 
    new_name=remove_ext(file)
    img=tiff.imread(path_+'/'+file)
    for i in range(7):
        if (i==0):
            name="Wetlands"
            new_path=os.path.join(path,name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sliced_img=slice(24,img)

            save_img(sliced_img,new_path,new_name)
        elif(i==1):
            name="Settlements"
            new_path=os.path.join(path,name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sliced_img=slice(54,img)

            save_img(sliced_img,new_path,new_name)
        elif(i==2):
            name="Water"
            new_path=os.path.join(path,name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sliced_img=slice(110,img)

            save_img(sliced_img,new_path,new_name)
        elif(i==3):
            name="Saltpan"
            new_path=os.path.join(path,name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sliced_img=slice(112,img)

            save_img(sliced_img,new_path,new_name)
        elif(i==4):
            name="Mangrove"
            new_path=os.path.join(path,name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sliced_img=slice(115,img)

            save_img(sliced_img,new_path,new_name)
        elif(i==5):
            name="Forest"
            new_path=os.path.join(path,name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sliced_img=slice(145,img)
            save_img(sliced_img,new_path,new_name)
        else:
            name="Open Land"
            new_path=os.path.join(path,name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sliced_img=slice(255,img)      
            save_img(sliced_img,new_path,new_name)                                    

        
            

print("Enter directory path")
path_=input()
ab_path=os.path.dirname(path_)
for file in os.listdir(path_):
    if file.endswith(".tif") or file.endswith(".tiff"):
        
        grayslice(file,path_)

    else:
        print("No Selection")

