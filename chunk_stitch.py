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


def factor(a):  # calculating factor to get chunk size

    for i in range(2, a - 1):
        if (a % i == 0 and (a / i) <= 200):
            return int(a / i)
        elif (a % i == 0 and isPrime(a / i) == True):  # if the factors are prime return it even if more than 5mb
            return int(a / i)
    return a


def isPrime(num):  # checking for prime
    if num > 1:

        # Iterate from 2 to n / 2
        for i in range(2, num):

            # If num is divisible by any number between
            # 2 and n / 2, it is not prime
            if (num % i) == 0:
                return False
                break
        else:
            return True

    else:
        return False


zN = z  # no change in z
xN = factor(x)  # slicing X
yN = factor(y)  # slicing Y
print(zN, yN, xN)


chunk_size = (zN,yN,xN)#setting chunk size
arr = da.from_array(data, chunks=chunk_size)#from_array generates chunks
print("chunked to size:" , zN*yN*xN*2/1000000 ,"MB each from: ",z*y*x*2/1000000,"MB")


i = 'A';

def info(chunk):
    global i
    if (chunk.shape[0] != 0):
        print("chunk shape", chunk.shape)
        io.imsave('newimages2/' + str(i) + '.tif', chunk)  # saving chunks
        i = chr(ord(i) + 1)
    return chunk


result_blocks = arr.map_blocks(info, dtype='uint16').compute(
    num_workers=1)  # map_blocks maps the info function across each chunk and stitches them back


print("Final Image shape", result_blocks.shape)
io.imsave('newimages2/FinalImage.tif', result_blocks)