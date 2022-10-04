import numpy as np
import os
import tifffile as tiff
from tqdm import tqdm


print('Enter Path')
path=input()
path=os.path.dirname(path)
print('\nEnter name of Save Directory')
save_dir=input()
final_dir=os.path.join(path,save_dir)
if not os.path.exists(final_dir):
    os.makedirs(final_dir)
print('Do you want to stack tiles or T3 matrix?')
print('\nPress t for tiles or press m for matrix')
yes_no=input()
if(yes_no=='t'):
    print("Please enter the total tiles to be stacked (Tiles in individual folders\n)")
    h1=input()
    h1=int(h1)
    h1+=h1

    for z in tqdm(range(1,h1),desc='Stacking Images'):    
        a=tiff.imread(path+'/T11/{}.tif'.format(z))
        b=tiff.imread(path+'/T12img/{}.tif'.format(z))
        c=tiff.imread(path+'/T12real/{}.tif'.format(z))
        d=tiff.imread(path+'/T13img/{}.tif'.format(z))
        e=tiff.imread(path+'/T13real/{}.tif'.format(z))
        f=tiff.imread(path+'/T22/{}.tif'.format(z))
        g=tiff.imread(path+'/T23img/{}.tif'.format(z))
        h=tiff.imread(path+'/T23real/{}.tif'.format(z))
        i=tiff.imread(path+'/T33/{}.tif'.format(z))
        wbar=np.dstack((a,b))
        wbar=np.dstack((wbar,c))
        wbar=np.dstack((wbar,d))
        wbar=np.dstack((wbar,e))
        wbar=np.dstack((wbar,f))
        wbar=np.dstack((wbar,g))
        wbar=np.dstack((wbar,h))
        wbar=np.dstack((wbar,i))
        reshape=wbar.transpose([2,0,1])
        tiff.imsave(final_dir+'/{}.tif'.format(z),reshape)
elif(yes_no=='m'):
    for z in tqdm(range(1,10),desc='Stacking Images'):    
        a=tiff.imread(path+'/T11.tif')
        b=tiff.imread(path+'/T12img.tif')
        c=tiff.imread(path+'/T12real.tif')
        d=tiff.imread(path+'/T13img.tif')
        e=tiff.imread(path+'/T13real.tif')
        f=tiff.imread(path+'/T22.tif')
        g=tiff.imread(path+'/T23img.tif')
        h=tiff.imread(path+'/T23real.tif')
        i=tiff.imread(path+'/T33.tif')
        wbar=np.dstack((a,b))
        wbar=np.dstack((wbar,c))
        wbar=np.dstack((wbar,d))
        wbar=np.dstack((wbar,e))
        wbar=np.dstack((wbar,f))
        wbar=np.dstack((wbar,g))
        wbar=np.dstack((wbar,h))
        wbar=np.dstack((wbar,i))
        reshape=wbar.transpose([2,0,1])
        tiff.imsave(final_dir+'/Stacked.tif',reshape)

else:
    print('Nothing was selected')

