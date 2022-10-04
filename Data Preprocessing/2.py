import numpy as np
import tifffile as tiff
from PIL import Image
Image.MAX_IMAGE_PIXELS=100000000000
import os
from tqdm import tqdm

def patch_img(img_path,tile_sz):
    img=Image.open(img_path)
    h,w,z=np.shape(img)
    patch_x=w//tile_size
    patch_y=h//tile_size
    total_patch=patch_x*patch_y
    pathname=os.path.dirname(img_path)
    name=os.path.splitext(os.path.basename(img_path))[0]
    final_path=os.path.join(pathname,name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    

    pbar=tqdm(total=total_patch,desc='Creating Tiles')
    left,top,right,bottom=0,0,0,0
    count=0
    for i in range(patch_y):
        for j in range(patch_x):
            left=j*tile_size+j 
            top=i*tile_size+i
            right=left+tile_size
            bottom=top+tile_size
            arr=tile_patch(img,left,top,right,bottom)
            count+=1
            pbar.update(1)
            tiff.imsave(final_path+'/{}.tif'.format(count),arr)
            print('\n{}.tif'.format(count),'has been saved.')
    
    return print('Job Done.')


def tile_patch(imag,l,t,r,b):
    cropped=imag.crop((l,t,r,b))
    convert_to_array=np.array(cropped)
    return convert_to_array




print("Please enter the directory path containing the T3 files in .tif format\n")
path_name=input()
print('\nEnter the tile size in pixels(Only enter the number, the tile will be a square of nxn)')
tile_size=input()
tile_size=int(tile_size)
dir_path=os.path.dirname(path_name)
for file in os.listdir(path_name):
    if file.endswith(".tif") :
        img_path=(dir_path+'/'+file)# Completes path name of the files
        read_img=tiff.imread(img_path)
        height,width,z=np.shape(read_img)
        print('H=',height,', W=',width)        
        total_patch_x=width//tile_size
        total_patch_y=height//tile_size
        total_patch=total_patch_x*total_patch_y
        print('A total of ',total_patch_x,'x',total_patch_y,'=',total_patch,'patches can be created')
        print('\n Enter y to proceed with tiling')
        yes_no=input()
        yes_no=str(yes_no)
        if(yes_no=='y'):
            patch_img(img_path,tile_size)
        else:
            print('No selection')

        



