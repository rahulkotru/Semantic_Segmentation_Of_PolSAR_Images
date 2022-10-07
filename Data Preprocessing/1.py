from PIL import Image
Image.MAX_IMAGE_PIXELS=100000000000000
from  tifffile import imsave
import gdal
import numpy as np
import os


def remove_ext(stringwe):#Function to extract file name from path
    print("Reading names")
    base=os.path.basename(stringwe)
    name=os.path.splitext(base)[0]
    return name

def convert(file_img,dirpath):#Function to convert .bin to .tif using GDAL
    
    new_name=remove_ext(file_img)#Get file name
    print("Getting File data on {}".format(new_name))
    fn=gdal.Open(dirpath+'/'+file_img) #Path of file with .bin/Gtiff extension
    fn_array=fn.ReadAsArray()

    if (os.path.exists(dirpath+'/'+'T3')==True):#Creating directory to save
        imsave(dirpath+'/'+'T3'+'/'+new_name+'.tif',fn_array)
        
        
    else:
        new_path=os.path.join(dirpath,'T3')
        os.makedirs(new_path)
        imsave(new_path+'/'+new_name+'.tif',fn_array) 





print("Please enter the directory path containing the T3 files in .bin format\n")
path_name=input()
dir_path=os.path.dirname(path_name)
for file in os.listdir(path_name):
    if file.endswith(".bin") :
        print("This is file name {}".format(file))
        convert(file,dir_path)
    else:
        print("Files not found")

12345