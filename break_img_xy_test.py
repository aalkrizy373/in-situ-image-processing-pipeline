from skimage import (
    exposure, io
)
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import dask
import dask.array as da

data = io.imread('10.tif')#path to the 3d image tif file
#printing shape and data type
print("shape: {}".format(data.shape))
print("dtype: {}".format(data.dtype))
