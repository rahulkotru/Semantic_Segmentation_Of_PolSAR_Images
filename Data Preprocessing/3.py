from PIL import Image
Image.MAX_IMAGE_PIXELS=100000000000000
from  tifffile import imsave
import numpy as np
import os



def flip(filename,path_of_file):
    img = Image.open(path_of_file+'/'+filename)
    Vert_flippedImage = img.transpose(Image.FLIP_TOP_BOTTOM)#Flips the image vertically
    #Hor_flippedImage = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_as_arr=np.asarray(Vert_flippedImage)#converts to array
    imsave(path_of_file+'/'+filename,img_as_arr)#saves the image




print("Please enter the directory path containing the T3 files to be flipped\n")
path_name=input()
dir_path=os.path.dirname(path_name)
for file in os.listdir(path_name):
    if file.endswith(".tif") :
        print("This is file name {}".format(file))
        flip(file,dir_path)
    else:
        print("Files not found")


