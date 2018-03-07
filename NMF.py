from libtiff import TIFF
from sklearn.decomposition import NMF  
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import sys
import os
import os.path
import shutil
import scipy.misc

image_src_dir = argv[1]
image_output_dir = argv[2]

if os.path.exists(image_output_dir):
    shutil.rmtree(image_output_dir)
if not os.path.exists(image_output_dir):
    os.mkdir(image_output_dir)

def find_last_point(file_path):
    position = 0
    temp = 0
    while temp != -1:
        temp = file_path.find(".")
        if temp != -1:
            position = temp
            file_path = file_path[ temp + 1:]
    return position

def check_image(root_dir):
    if not os.path.isdir(root_dir):
        return

    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        # handle output path of dir and file
        out_path = image_output_dir + path[9:]

        if os.path.isfile(path):
            # handle output file name
            point_position = find_last_point(out_path)
            if not set_img_type == "":
                out_path = out_path[0:point_position + 1] + 'txt'
            # to open a tiff file for reading:
            tif=TIFF.open(path, mode='r')
            # to read an image in the currect TIFF directory and return it as numpy array:
            image = tif.read_image()
            #Using NMF to class pixel into two groups
            n_components = 2
            estimator = NMF(n_components = n_components, init = 'random', tol=5e-3)    
            W = estimator.fit_transform(image)
            H = estimator.components_
            new_image = np.dot(W,H)
            #resize the image
            #image_shape = (64, 64)
            #out_image=transform.resize(new_image,image_shape)
            matplotlib.image.imsave(out_path+, new_image)

        if os.path.isdir(path):
            # make dir in image\output
            if not os.path.exists(out_path):
                os.mkdir(out_path)
                print out_path
            check_image(path)

check_image(image_src_dir)



