# This source file was built based on example.py in the datasets. 
# for more info see:
#
# - http://neurofinder.codeneuro.org
# - https://github.com/codeneuro/neurofinder
#
# requires three python packages
#
# - numpy
# - scipy
# - matplotlib
#

import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from scipy.misc import imread
import PIL
from PIL import Image
from glob import glob
import numpy as np
import math

# load the images
files = sorted(glob('images/image000**.tiff'))
imgs = array([imread(f) for f in files])
dims = imgs.shape[1:]

# load the regions (training data only)
with open('regions/regions.json') as f:
    regions = json.load(f)

def tomask(coords):
    mask = zeros(dims)
    mask[list(zip(*coords))] = 1
    return mask

masks = array([tomask(s['coordinates']) for s in regions])

image = imgs.sum(axis=0)
mask = masks.sum(axis=0)



def avg_noise_redction(image):
    avg_img = np.zeros_like(image)
    for i in range(len(image)-2):
        for j in range(len(image[0])-2):
            avg_arr = []
            avg_arr.extend((image[i][j],image[i,j+1],image[i,j+2],
                            image[i+1][j],image[i+1][j+1],image[i+1][j+2],
                            image[i+2][j],image[i+2][j+1],image[i+2][j+2]))
            avg_pix = math.ceil(np.mean(avg_arr))
            avg_img[i+1][j+1] = avg_pix

    return avg_img

def median_filtering(image):
    med_img = np.zeros_like(image)
    for i in range(len(image)-2):
        for j in range(len(image[0])-2):
            avg_arr = []
            avg_arr.extend((image[i][j],image[i,j+1],image[i,j+2],
                            image[i+1][j],image[i+1][j+1],image[i+1][j+2],
                            image[i+2][j],image[i+2][j+1],image[i+2][j+2]))
            med_pix = np.median(avg_arr)
            med_img[i+1][j+1] = med_pix

    return med_img

theImage = Image.fromarray(  np.uint16(   avg_noise_redction( imgs[0] )))
theImage_median =  Image.fromarray(  np.uint16(   median_filtering( imgs[0] )))
theMask =  Image.fromarray( ( np.uint8(mask) )*255)

 
theImage.show()
theImage_median.show()
theMask.show()


# show the outputs
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(imgs.sum(axis=0), cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(masks.sum(axis=0), cmap='gray')
# plt.show()