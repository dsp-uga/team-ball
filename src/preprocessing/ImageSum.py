from src.preprocessing.preprocessor import preprocessor
import numpy as np
import cv2
from numpy import array, zeros
import os
from glob import  glob
from scipy.misc import imread
import json

class ImageSum ( preprocessor ):

    def __init__(self , exportPath, trainingPath , testPath , images_size, importPath = None ):
        self.exportPath = exportPath
        self.traininPath = trainingPath
        self.testPath = testPath
        self.image_size = images_size
        self.importPath = importPath

    def loadSample ( self, path ):
        """
        this funtion loads the images in the sample as one output image

        :param paht: path ot the sample this has to be in glob format describing path to TIFF images
        :return: returns one image which is the aggregated version of all images in the sample
        """

        # read image files
        files = sorted(glob(path))
        imgs = array([imread(f) for f in files])

        # merge files in to one image
        image = imgs.sum(axis=0)

        image = self.change_size( image, self.image_size )

        print(image.shape)

        return image

    def load_from_files(self):
        """
        loads images from previously preprocessed files. 
        :return: a treplet of  ( train_x, train_y, test_dic )
        """

        if( self.importPath ):
            train_x  = np.load( os.path.join( self.importPath , "X_train.npy" ) )
            train_y = np.load(os.path.join( self.importPath , "Y_train.npy" ))

            test_dic = {}

            for file in  glob( os.path.join( self.importPath, "*.test.npy" ) ):
                test_dic[ file.replace('.test.npy','').replace('neurofinder.','') ]  = np.load( file )

            return train_x, train_y, test_dic

        return None

    def preprocess(self):
        """
        this funciton preopricess the imagaes into
        :return:
        """
        train_x = []  # None #np.array([])
        train_y = []  # None # np.array([])


        # # check if there is an improt path supplied, load information from
        # if( self.importPath ):
        #     return  self.load_from_files()

        # create the trainig set
        if( self.traininPath ):
            for sample in os.listdir(self.traininPath):
                images_glob_path = os.path.join( self.traininPath,    sample + "/images/*.tiff")
                mask_json_path = os.path.join( self.traininPath, sample + '/regions/regions.json')

                images = self.loadSample( images_glob_path )

                train_x.append( list( images))

                mask = self.loadRegions( mask_json_path, self.image_size  )

                train_y.append( list( mask ) )

        # create the test set
        test_dic = {}
        if( self.testPath ):
            for sample in os.listdir(self.testPath):
                images_glob_path = os.path.join( self.testPath,    sample + "/images/*.tiff")

                images = self.loadSample(images_glob_path)

                test_dic[ sample ] = images

        train_x = array(train_x)
        train_y = array(train_y)

        if( self.exportPath ):
            # save train and test files
            np.save( os.path.join( self.exportPath , 'X_train.npy'), train_x)
            np.save( os.path.join( self.exportPath , 'Y_train.npy'), train_y)

            #save the testsamples
            for key in test_dic:
                np.save( os.path.join( self.exportPath , key+".test.npy" ) , test_dic[key])

        return train_x , train_y , test_dic

