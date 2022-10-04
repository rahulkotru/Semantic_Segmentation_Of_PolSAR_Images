import numpy as np
import tifffile as tiff



def stitch(start,end,name):
    

    for i in range(start,end):
        img1=tiff.imread('C:/Users/rrkot/OneDrive/Desktop/New Matrix/Del/GT/{}.tif'.format(i))#Please enter the directory containing the predicted tiles
        if i==start:
            
            d=img1
        else:

        
            c=np.concatenate((d,img1),axis=1)

            d=c     

    tiff.imsave('C:/Users/rrkot/OneDrive/Desktop/New Matrix/Del/GT/GT{}.tif'.format(i))#Enter the save directory for horizontal stitch




print("Enter total images")#Total images in the directory
total=input()
total=int(total)
print("Enter total images per row")#Number of images needed per row(Width of the recombined image)
width=input()
width=int(width)
height=int(total/width)#Total height possible of the recombined image
z=1
new_start=0
for j in range(1,height+1):

    
    
    if j==1:
        start=j
        end=start+width
        print('Calling Stitch for Start=',start,'End=',end)
        stitch(start,end,j)
        new_start=end

    else:
        start=new_start
        end=start+width
        print('Calling Stitch for Start=',start,'End=',end)
        stitch(start,end,j)
        new_start=end


'''
jj=tiff.imread('C:/Users/rrkot/OneDrive/Desktop/Semantic Results/Mumbai/Pred/4.tif')
g=np.shape(jj)
g
tiff.imsave('C:/Users/rrkot/OneDrive/Desktop/New Matrix/Ground Truth/1.tif',jj)
'''