
# Upload Images (Remove for non-colab env)

from typing import Mapping, Union

from starfish.image import ApplyTransform
from starfish.image import LearnTransform
import matplotlib
from starfish.util.plot import diagnose_registration
from starfish.types import Axes

from starfish.core.types import Coordinates, CoordinateValue, Axes
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import dask
import dask.array as da
from skimage import (
    io
)
# Enter the common name and number of images you want to enter

#Example:
#for Max_base1, Max_base2, Max_base3 - --> name: Max_base and number: 3


name = input("Enter common name: ")
number = int(input("Enter number of images: "))


# Read the reference image(Last image of series)


data2 = io.imread(str(name) + str(number) + '.tif')  # path to the image tif file
# printing shape and data type
print("shape: {}".format(data2.shape))
print("ndim: {}".format(data2.ndim))
from typing import Mapping, Union
from starfish import ImageStack

if (data2.ndim == 3):  # for 2d channeled
    y1, x1, c1 = data2.shape  # getting image shape
    print(data2.shape)
    print(data2.ndim)
    print(data2.dtype)
    # data=data=np.reshape(data, (y,x,c))
    arr2 = np.split(data2, indices_or_sections=c1, axis=2)
    for ch in range(0, c1):
        arr2[ch] = np.reshape(arr2[ch], (y1, x1))  # reshaping each channel to single channels

    arr2 = np.array(arr2)
    print(arr2.shape)

    c1, y1, x1 = arr2.shape
    num_r = 1
    num_c = c1
    z1 = 1

    dot = np.resize(arr2, (num_r, num_c, z1, y1, x1))  # reshaping to 5D
    print(dot.shape)
    print(dot.ndim)
else:  # for 3d channeled

    z1, y1, x1, c1 = data2.shape

    arr2 = np.split(data2, indices_or_sections=c1, axis=3)

    for ch in range(0, c1):
        arr2[ch] = np.reshape(arr2[ch], (z1, y1, x1))  # reshaping each channel to single channels

    arr2 = np.array(arr2)
    print(arr2.shape)

    c1, z1, y1, x1 = arr2.shape
    num_r = 1
    num_c = c1

    dot = np.resize(arr2, (num_r, num_c, z1, y1, x1))  # reshaping to 5D
    print(dot.shape)
    print(dot.ndim)


# Loop and read all the images user wants to enter and align them W.R.T. the reference image
for i in range(1, (number + 1)):
    print(str(name) + str(i) + '.tif')
    data = io.imread(str(name) + str(i) + '.tif')
    if (data.ndim == 3):  # for 2d channeled
        y, x, c = data.shape  # getting image shape
        print(data.shape)
        print(data.ndim)
        print(data.dtype)
        # data=data=np.reshape(data, (y,x,c))
        arr = np.split(data, indices_or_sections=c, axis=2)
        for ch in range(0, c):
            arr[ch] = np.reshape(arr[ch], (y, x))  # reshaping each channel to single channels

        arr = np.array(arr)
        print(arr.shape)

        c, y, x = arr.shape
        num_r = 1
        num_c = c
        z = 1
        synthetic_data = np.resize(arr, (num_r, num_c, z, y, x))
        print(synthetic_data.shape)
        print(synthetic_data.ndim)
    else:  # for 3d channeled

        z, y, x, c = data.shape

        arr = np.split(data, indices_or_sections=c, axis=3)
        for ch in range(0, c):
            arr[ch] = np.reshape(arr[ch], (z, y, x))  # reshaping each channel to single channels

        arr = np.array(arr)
        print(arr.shape)

        c, z, y, x = arr.shape
        num_r = 1
        num_c = c
        synthetic_data = np.resize(arr, (num_r, num_c, z, y, x))
        print(synthetic_data.shape)
        print(synthetic_data.ndim)

    stacks = ImageStack.from_numpy(synthetic_data)  # converting numpy array to starFish format
    print(stacks)

    dott = ImageStack.from_numpy(dot)
    print(dott)
    imgs = stacks
    print(imgs)
    dots = dott.reduce({Axes.ZPLANE, Axes.CH}, func="max")  # storing MIP of reference image as dots
    print(dots)
    projected_imgs = imgs.reduce({Axes.ZPLANE, Axes.CH}, func="max")  # storing MIP of image to be aligned
    print(projected_imgs)
    learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND,
                                                   upsampling=1000)  # set reference stack=dots and axes to apply as rounds
    transforms_list = learn_translation.run(
        projected_imgs)  # learn the translation to be done based on MIP of reference wrt MIP of image to be aligned
    transforms_list.to_json('transforms_list.json')  # save json file
    print(transforms_list)

    warp = ApplyTransform.Warp()  # call apply transform function of skimage
    registered_imgs = warp.run(imgs, transforms_list=transforms_list,
                               in_place=False)  # run for the whole image stack to be aligned
    diagnose_registration(registered_imgs.reduce({Axes.ZPLANE, Axes.CH}, func="max"), {Axes.ROUND: 0})
    numpydata = registered_imgs
    numpydata.to_multipage_tiff(str(name) + str(i) + '_aligned.tiff')  # save as tiff file
