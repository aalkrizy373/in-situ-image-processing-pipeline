from skimage import (
     io
)
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import dask
import dask.array as da

data = io.imread('/Users/amr/Documents/Asrar/PycharmProjects/Church Lab/img_chunk/3D Images/3mem_3m0_1-1 ROI 1_crop_crop_crop_Cy5.tif')#loading tif image
z,y,x=data.shape;#shape outputs 3 values
dim=data.ndim#dimension
print("shape: ",z,y,x)
print("dtype: ",(data.dtype))#datatype
print("dtype: ",dim)

def calc(a):#calculating factor to get chunk size

    if(a%2==0):
        return int(a/2)
    else:
        return int(a/2)+1



zN=z#no change in z
xN=calc(x)#slicing X
yN=calc(y)#slicing Y
print(zN,yN,xN)

chunk_size = (zN,yN,xN)#setting chunk size
arr = da.from_array(data, chunks=chunk_size)#from_array generates chunks
print("chunked to size:" , zN*yN*xN*2/1000000 ,"MB each from: ",z*y*x*2/1000000,"MB")

i = 'A';


def info(chunk):
    global i
    if (chunk.shape[0] != 0):
        print("saving chunk", i)
        io.imsave('newimages2/' + str(i) + '.tif', chunk)  # saving chunks
        i = chr(ord(i) + 1)
    return chunk


# result_blocks = arr.map_blocks(info,dtype='uint16').compute(num_workers=1)  #map_blocks maps the info function across each chunk and stitches them back
result_overlap = arr.map_overlap(info, depth=(6, 6, 6), boundary='reflect', dtype='uint16').compute(num_workers=1)


print("Final Image shape", result_overlap.shape)
io.imsave('newimages2/FinalImage.tif', result_overlap)