# Upload Image of any channel

from skimage import (
     io
)
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import dask
import dask.array as da

data = io.imread('3mem_3m0_1-1 ROI 1_crop_crop_crop.tif')#path to the 3d image tif file
#printing shape and data type
print("shape: {}".format(data.shape))
print("dtype: {}".format(data.dtype))
dim=data.ndim


# Check if single channeled

if (dim==3):#if single channel

  z3,y3,x3=data.shape
  data=np.reshape(data, (z3,y3,x3,1))  #make it multi channel format

# Split channels

z,y,x,c=data.shape
arr=np.split(data,indices_or_sections= c,axis=3)#split channels, indices: number of channels, axis: 3(last)


for ch in range(0,c):

 arr[ch]=np.reshape(arr[ch], (z,y,x))  #reshaping each channel to single channels
 print(arr[ch].shape)
 print(arr[ch].ndim)
 print(arr[ch].dtype)


# Calculate chunk size

def calc(a):#calculating the chunk size by dividing by 2
  if(a%2==0):
    return int(a/2)#for even length return length by 2
  else:
    return int(a/2)+1 #for odd length round up

Zn=z#new z value same as original
Yn=calc(y)#new y value calulated as y by 2
Xn=calc(x)#new x value calculated as x by 2
print(Zn,Yn,Xn)

# Perform decon for each channel

import numpy as np
import matplotlib.pyplot as plt
from flowdec.nb import utils as nbutils
from flowdec import psf as fd_psf
from flowdec import data as fd_data
from scipy import ndimage
import dask
import dask.array as da
import tensorflow as tf
from flowdec.restoration import RichardsonLucyDeconvolver



import operator
def cropND(img, bounding):#cropping image
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))#dividing lengths by 2
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))#make slices of given shape(bounding)
    return img[slices]
def decon2(chunk):#deconvolution function
  if(chunk.shape[0]!=0):#to eliminate initializing empty chunk

    x1,y1,z1=chunk.shape
    #print(x,y,z)
    psf = fd_psf.GibsonLanni(
    pz=0.4, #Z value
    na=1.4, #numerical aperture
    m=40,  #magnification
    ns=1.33,    #specimen refractive index (RI)
    ni0=1.5,    # immersion medium RI design value
    tg0 =200, # microns, coverslip thickness design value
    ti0=100, # microns, working distance (immersion medium thickness) design value
    tg=200,     # microns, coverslip thickness experimental value
    res_lateral=5.5, # X/Y resolution
    res_axial=0.16,     # Axial resolution
    wavelength=.700,  # Emission wavelength
    num_basis=300, #the number of approximation basis
    num_samples=1000, #the number of sampling to determine the basis
    size_x=x,
    size_y=y,
    size_z=z).generate()
    #print(psf)

    psf = cropND(psf, (x1-12,y1-12,z1-12))#cropping kernel with chunk shape, padding 6,6,6 is used hence for each axis 6 is padded on each side thus -12 is used to get non-padded value. kernel cannot have overlap, only chunk has
    algo = RichardsonLucyDeconvolver(3, pad_mode="2357", pad_min=(6,6,6))#pad_mode->default padding type, pad_min->overlap for each axis, that can be changed


    print("Applying deconvolution...")
    res = algo.initialize().run(fd_data.Acquisition(data=chunk, kernel=psf), 1)#running the LR algo on chunk with cropped kernel


    return res.data#returning output

Fresult=[]#list to store each decon channel
for ch in range(0,c):
# chunked dask array
 das = da.from_array(arr[ch], chunks=(Zn,Yn,Xn))#Xn,Yn,Zn are new shapes where X and Y are divided by 2 and Z axis is same
# kernel cropped to chunk size
 print("Channel scliced")


 result = das.map_overlap(decon2,depth=(6,6,6), boundary='reflect', dtype='float64').compute(num_workers=1)#depth: overlap (can be chnaged), boundary->default, dtype->data type of image, num_workers-> number of CPU processors to be used.
 print(result.shape)
 Fresult.append(result)#inserting result to list
 print(len(Fresult))
Fresult=np.array(Fresult)#making the list an array
print(Fresult.shape)

# Line up channels

if(c>1):#if multi channel
 resultant=np.stack(Fresult, axis=-1)#line up the channels to last axis
else:#if single channel keep as it is
  resultant= result
print('Result shape (z, y, x, c):', resultant.shape)
print('Result dtype:', resultant.dtype)


# Print result array


print(resultant)


# Save decon image


resultant=resultant.astype('uint16')#type casting result

io.imsave('Decon_image01.tif', resultant)#saving image