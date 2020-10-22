'''
import image and check for shape, dimension and datatype
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import dask
import dask.array as da
from skimage import (
     io
)

data = io.imread('/Users/amr/Documents/Asrar/PycharmProjects/Church Lab/img_chunk/3D Images/3mem_3m0_1-1 ROI 1_crop_crop_crop_Cy5.tif')#import file path
z,y,x=data.shape#getting image shape
print(z,y,x)
print(data.ndim)
print(data.dtype)

'''
Calculating Chunk Shape
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import dask
import dask.array as da
from skimage import (
     io
)

data = io.imread('/Users/amr/Documents/Asrar/PycharmProjects/Church Lab/img_chunk/3D Images/3mem_3m0_1-1 ROI 1_crop_crop_crop_Cy5.tif')#import file path
z,y,x=data.shape#getting image shape
print(z,y,x)
print(data.ndim)
print(data.dtype)

'''
Performing Deconvolution
'''
import numpy as np
import matplotlib.pyplot as plt
from flowdec.nb import utils as nbutils
from flowdec import data as fd_data
from scipy import ndimage
import dask
import dask.array as da
import tensorflow as tf
from flowdec.restoration import RichardsonLucyDeconvolver

kernel = np.zeros_like(data.astype(
    'float32'))  # creating decon kernel(float32 is the commonly used datatype of kernel because else result will return infinite)
for offset in [0, 1]:
    kernel[tuple((np.array(kernel.shape) - offset) // 2)] = 1
kernel = ndimage.gaussian_filter(kernel,
                                 sigma=2.)  # making gaussian kernel ---> sigma value can be changed to get different decon results

import operator
def cropND(img, bounding):  # cropping image
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))  # dividing lengths by 2
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))  # make slices of given shape(bounding)
    return img[slices]


# chunked dask array
arr = da.from_array(data,
                    chunks=(Zn, Yn, Xn))  # Xn,Yn,Zn are new shapes where X and Y are divided by 2 and Z axis is same


# kernel cropped to chunk size

def decon2(chunk):  # deconvolution function
    if (chunk.shape[0] != 0):  # to eliminate initializing empty chunk

        x1, y1, z1 = chunk.shape

        cropped_kernel = cropND(kernel, (x1 - 12, y1 - 12,
                                         z1 - 12))  # cropping kernel with chunk shape, padding 6,6,6 is used hence for each axis 6 is padded on each side thus -12 is used to get non-padded value. kernel cannot have overlap, only chunk has
        algo = RichardsonLucyDeconvolver(data.ndim, pad_mode="2357", pad_min=(
        6, 6, 6))  # pad_mode->default padding type, pad_min->overlap for each axis, that can be changed

        print("Applying deconvolution...")
        res = algo.initialize().run(fd_data.Acquisition(data=chunk, kernel=cropped_kernel),
                                    1)  # running the LR algo on chunk with cropped kernel

        return res.data  # returning output


result2 = arr.map_overlap(decon2, depth=(6, 6, 6), boundary='reflect', dtype='float32').compute(
    num_workers=1)  # depth: overlap (can be changed), boundary->default, dtype->data type of image, num_workers-> number of CPU processors to be used.

'''
Print Deconvolted Image
note: cmap:viridis
'''
nbutils.plot_rotations(result2)#print decon images
result=result2.astype('uint16')#changing back to original dtype
nbutils.plot_rotations(result)#plotting images

'''
Save Image
'''
from skimage import io
io.imsave('Deconimage.tif', result)
#save decon image

print(result.dtype)#print final image details
print(result.shape)
print(result.ndim)