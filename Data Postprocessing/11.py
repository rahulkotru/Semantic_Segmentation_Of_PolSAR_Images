import numpy as np
import cv2
import tifffile as tiff
import os
from PIL import Image

def unique_colours(arr):
    flat=arr.flatten()
    print("Analysing ",(len(flat))," pixels....")
    s=set()
    res=0
    for j in range(len(flat)):
        if(flat[j] not in s):
            s.add(flat[j])
            res+=1
    print("Color List= {se} , and total distinct classes in image are {reys}".format(se=s,reys=res))
    return s,res

def graycode(arr):
    gray_image=cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
    return gray_image 

def color_threshold(array):
    (x,y)=array.shape
    arr=array.flatten()
    for u in range(len(arr)):
        if(arr[u]>=2 and arr[u]<40):
            arr[u]=24#Wetland
        elif(arr[u]>41 and arr[u]<58):
            arr[u]=54#Settlement
        elif(arr[u]>=59 and arr[u]<111):
            arr[u]=110#Water
        elif(arr[u]>111 and arr[u]<113):
            arr[u]=112#Saltpan
        elif(arr[u]>114 and arr[u]<120):
            arr[u]=115#Mangrove
        elif(arr[u]>121 and arr[u]<160):
            arr[u]=145#Forest
        else:
            arr[u]=255#Open Land
    return arr.reshape(x,y)

def remove_ext(stringwe):
    base=os.path.basename(stringwe)
    name=os.path.splitext(base)[0]
    return name

def save_img(arr,path,name):
    print("Saving Images...\n")
    new_path=os.path.join(path,'Saved Images')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    tiff.imsave(new_path+'/{}.tif'.format(name),arr)

print("Enter the path of the directory\n")
path=input()
#directory=os.fsencode(path)

for file in os.listdir(path):
    #filename=os.fsdecode(file)
    if file.endswith(".tif") or file.endswith(".tiff"):

        print("Proceeding with Image Analysis")
        print(file)

        image=tiff.imread(path+'/'+file).transpose([1,2,0])

        gray_img=graycode(image)
        name=remove_ext(file)
        changed_image=color_threshold(gray_img)
        sett,distinct_colours=unique_colours(changed_image)
        save_img(changed_image,path,name)
        
    else:
        print("Invalid Directory")

'''
imgr=tiff.imread('C:/Users/rrkot/OneDrive/Desktop/Semantic Results/Mumbai/trial/Prediction.tif')
rgb_image = imgr.transpose([1,2,0])
tiff.imsave('C:/Users/rrkot/OneDrive/Desktop/Semantic Results/Mumbai/trial/41.tif',rgb_image)
'''