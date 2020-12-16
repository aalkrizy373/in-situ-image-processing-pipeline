from starfish.core.types import Coordinates, CoordinateValue, Axes
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import dask
import dask.array as da
from skimage import (
    io
)

# Input common name, number of images, minimum spot size(preferable less then 10) and z_max for dummy z planes to make all z values same

name = input("Enter common name: ")
number = int(input("Enter number of images: "))
m_size = int(input("Enter minimum area of spot: "))
z_max = int(input("Enterthe max z value: "))

# Store images in list

img = []

for j in range(1, (number + 1)):
    print(str(name) + str(j) + '.tif')
    ab = io.imread(str(name) + str(j) + '.tif')
    print(ab.shape)
    img.append(ab)

# Read images and reshape to c,z,y,x

image = []
for i in range(0, len(img)):
    data = img[i]

    if (data.ndim == 3):  # for 2d channeled
        y, x, c = data.shape  # getting image shape

        arr = np.split(data, indices_or_sections=c, axis=2)
        for ch in range(0, c):
            arr[ch] = np.reshape(arr[ch], (1, y, x))  # reshaping each channel to single channels

        arr = np.array(arr)
        print(arr.shape)

        c, z, y, x = arr.shape

    else:  # for 3d channeled

        z, y, x, c = data.shape
        zero = np.zeros((z_max, y, x, c), dtype='uint16')
        z = 0
        x = 0
        y = 0
        c = 0

        zero[z:z + data.shape[0], y:y + data.shape[1], x:x + data.shape[2],
        c:c + data.shape[3]] = data  # make 3d image of same number of planes
        z, y, x, c = zero.shape

        arr = np.split(zero, indices_or_sections=c, axis=3)
        for ch in range(0, c):
            arr[ch] = np.reshape(arr[ch], (z, y, x))  # reshaping each channel to single channels

        arr = np.array(arr)
        print('arr', arr.shape)

        c, z, y, x = arr.shape
    image.append(arr)

# Making list of images as an array of images to make it multi round

import skimage

image = np.array(image)
r, c, z, y, x = image.shape
print(image.shape)

# make the image array to image stack of starfish format

from starfish import ImageStack

stack = ImageStack.from_numpy(image)  # converting numpy array to starFish format
print(stack)

# Create 3 pattern codebook

import numpy as np
from starfish import Codebook

code = np.zeros((3, 5, 3), dtype=np.uint8)  # create array of 3 pattern, 5 round, 3 channel
code[0, 0, 1] = 1
code[0, 1, 0] = 1
code[0, 1, 1] = 1
code[0, 2, 1] = 1
code[0, 3, 0] = 1
code[0, 3, 1] = 1
code[0, 4, 1] = 1

code[1, 0, 0] = 1
code[1, 1, 0] = 1
# code[1,1,1]=1
code[1, 2, 0] = 1
code[1, 3, 1] = 1
# code[1,3,1]=1
code[1, 4, 0] = 1

code[2, 0, 0] = 1
code[2, 1, 0] = 1
code[2, 1, 1] = 1
code[2, 2, 0] = 1
code[2, 3, 0] = 1
code[2, 3, 1] = 1
code[2, 4, 0] = 1
sd = Codebook.from_numpy(['A', 'B', 'C'], n_channel=3, n_round=5, data=code)
print(sd)

# from starfish import Codebook
# sd = Codebook.synthetic_one_hot_codebook(n_round=5, n_channel=3, n_codes=100)
# print(sd)

# Pixel based decoding done on the codebook

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from starfish import data, FieldOfView, display
from starfish.image import Filter
from starfish.spots import DetectPixels
from starfish.types import Axes, Features, Levels

imgs = stack

##### filter data if required #####

# ghp = Filter.GaussianHighPass(sigma=1)
# dpsf = Filter.DeconvolvePSF(num_iter=15, sigma=2, level_method=Levels.SCALE_SATURATED_BY_CHUNK)
# glp = Filter.GaussianLowPass(sigma=1)
# ghp.run(imgs, in_place=True)
# dpsf.run(imgs, in_place=True)
# glp.run(imgs, in_place=True)

##########################################

# scale data with user-defined factors to normalize images. For this data set, the scale factors
# are stored in experiment.json.
print(imgs)

# Decode with PixelSpotDecoder
psd = DetectPixels.PixelSpotDecoder(
    codebook=sd,
    metric='euclidean',  # distance metric to use for computing distance between a pixel vector and a codeword
    norm_order=2,  # the L_n norm is taken of each pixel vector and codeword before computing the distance. this is n
    distance_threshold=0.6576,
    # has to be above 0.5175 ***** INCREASE VALUE IN CASE OF ERROR (due to low intensity of image if it fails to find the spot present) *****
    magnitude_threshold=0.77e-5,  # discard any pixel vectors below this magnitude
    min_area=m_size,  # do not call a 'spot' if it's area is below this threshold (measured in pixels)
    max_area=np.inf,  # do not call a 'spot' if it's area is above this threshold (measured in pixels)
)
initial_spot_intensities, prop_results = psd.run(imgs)

# filter spots that do not pass thresholds
spot_intensities = initial_spot_intensities.loc[initial_spot_intensities[Features.PASSES_THRESHOLDS]]

# View labeled image after connected componenet analysis

# Make the arrays for 5 intensity excel sheets and 1 target excel sheet

x_axis = []
y_axis = []
z_axis = []
target = []
r1c1 = []
r1c2 = []
r1c3 = []
r2c1 = []
r2c2 = []
r2c3 = []
r3c1 = []
r3c2 = []
r3c3 = []
r4c1 = []
r4c2 = []
r4c3 = []
r5c1 = []
r5c2 = []
r5c3 = []
for i in range(0, spot_intensities.features.size):
    x_axis.append(spot_intensities.coords['x'][i].values)
    y_axis.append(spot_intensities.coords['y'][i].values)
    z_axis.append(spot_intensities.coords['z'][i].values)
    target.append(spot_intensities.coords['target'][i].values)
    r1c1.append(spot_intensities[i][0][0].values)
    r1c2.append(spot_intensities[i][0][1].values)
    r1c3.append(spot_intensities[i][0][2].values)
    r2c1.append(spot_intensities[i][1][0].values)
    r2c2.append(spot_intensities[i][1][1].values)
    r2c3.append(spot_intensities[i][1][2].values)
    r3c1.append(spot_intensities[i][2][0].values)
    r3c2.append(spot_intensities[i][2][1].values)
    r3c3.append(spot_intensities[i][2][2].values)
    r4c1.append(spot_intensities[i][3][0].values)
    r4c2.append(spot_intensities[i][3][1].values)
    r4c3.append(spot_intensities[i][3][2].values)
    r5c1.append(spot_intensities[i][4][0].values)
    r5c2.append(spot_intensities[i][4][1].values)
    r5c3.append(spot_intensities[i][4][2].values)

import pandas as pd

dict = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'target': target}
df1 = pd.DataFrame(dict)  # making pandas dataframe from dictionary
df1.to_csv('target.csv')
dict = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'channel1': r1c1, 'channel2': r1c2, 'channel3': r1c3}
df2 = pd.DataFrame(dict)
df2.to_csv('Intensity_table1.csv')
dict = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'channel1': r2c1, 'channel2': r2c2, 'channel3': r2c3}
df3 = pd.DataFrame(dict)
df3.to_csv('Intensity_table2.csv')
dict = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'channel1': r3c1, 'channel2': r3c2, 'channel3': r3c3}
df4 = pd.DataFrame(dict)
df4.to_csv('Intensity_table3.csv')
dict = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'channel1': r4c1, 'channel2': r4c2, 'channel3': r4c3}
df5 = pd.DataFrame(dict)
df5.to_csv('Intensity_table4.csv')
dict = {'x': x_axis, 'y': y_axis, 'z': z_axis, 'channel1': r5c1, 'channel2': r5c2, 'channel3': r5c3}
df6 = pd.DataFrame(dict)
df6.to_csv('Intensity_table5.csv')

# View resultant image

plt.imshow(prop_results.label_image[0])

print(imgs)

# Save image

print(prop_results.label_image.astype('float32'))
io.imsave('multiround-finalimage.tif', prop_results.label_image.astype('float32'))
