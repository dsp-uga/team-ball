"""Abstract calss for preprocessors"""

import json
import numpy as np
import os
from numpy import array, zeros

class preprocessor:

    def __init__(self):
        self.data= None

    def loadSample(self, path):

        return  None;

    def preprocess (self):
        return None

    def loadRegions ( self , regionPath  , dims):
        """
        this function loads the region file into a mask image
        :param regionPath: path to the region file
        :param dims: target dimentions
        :return: returns the mask image
        """
        def tomask(coords):
            """
            this function turns the region pixels into a mask
            :param coords: the input array
            :return: returns the nparray representing the mask as image
            """

            mask = zeros(dims)
            mask[list(zip(*coords))] = 1
            return mask


        with open(regionPath) as f:
            regions = json.load(f)

        masks = array([tomask(s['coordinates']) for s in regions])

        mask = masks.sum(axis=0)

        return mask


    def change_size ( self , source , target_dim ):
        """
        this function up samples the image into the given dimensions by padding the image with zeros until it
        fits the goven size.

        :param source: the array to resize
        :param target_dim:  the target size
        :return: returns the resized image
        """

        dims = source.shape
        if dims[0] != target_dim[0] or dims[1] != target_dim[1]:
            """
            if the source is not in the desired format change it to the desired size  
            """
            temp_mask = np.zeros((target_dim[0], target_dim[1]))

            temp_mask[:dims[0], :dims[1]] = source

            return temp_mask

        else :
            # if the image is already in the dessired size, return it.
            return source