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
import cv2
import os
import argparse

# load the images
# description = 'Tiff image preprocessing. This program converts the images into numpy array files'
#
# parser = argparse.ArgumentParser(description=description, add_help='How to use', prog='python main.py <options>')
#
# parser.add_argument("-t", "--dataset", default="data/X_train_vsmall.txt",
#                     help="Path to text file containing the documents in the training set"
#                          "[DEFAULT: \"data/X_train_vsmall.txt\"]")

TRAIN_DIR = os.path.abspath("../../data/train/")
TEST_DIR = None

def tomask(coords):
    mask = zeros(dims)
    mask[list(zip(*coords))] = 1
    return mask

train_x = []#None #np.array([])
train_y = []#None # np.array([])
i = 0
for sample in os.listdir( TRAIN_DIR ):
    images_glob_path =  sample + "/images/*.tiff"
    mask_json_path = sample + '/regions/regions.json'
    # read image files
    files = sorted(glob(  os.path.join( TRAIN_DIR, images_glob_path )   ))
    imgs = array([imread(f) for f in files])
    dims = imgs.shape[1:]

    with open(  os.path.join(TRAIN_DIR , mask_json_path) ) as f:
        regions = json.load(f)

    masks = array([tomask(s['coordinates']) for s in regions])

    image = imgs.sum(axis=0)
    mask = masks.sum(axis=0)

    if dims[0]!= 512 or dims[1]!= 512:
        temp_image = np.zeros((512,512))
        temp_mask = np.zeros((512,512))

        temp_image[:image.shape[0], :image.shape[1]] = image
        temp_mask[:mask.shape[0], :mask.shape[1]] = mask

        image=temp_image
        mask = temp_mask

    print( image.shape )
    image =list( image )
    mask = list( mask )

    train_x.append( image )
    train_y.append(mask)
    del imgs
    del masks
    # if( train_x is None ):
    #     train_x = image
    #     train_y = mask
    # else:
    #     train_x = np.vstack( (train_x, image) )
    #     train_y = np.vstack( (train_y, mask) )
    i+=1
    print( i )
    # if i==5 :
    #     break;

train_x =  array(train_x)
train_y = array(train_y)

print (train_x.shape)
print (train_y.shape)

np.save('X_train.npy',  train_x)
np.save('Y_train.npy' , train_y)

