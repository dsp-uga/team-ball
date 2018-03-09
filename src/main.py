"""
this file contains the main runner for the project
"""

import argparse
import sys
import os
import logging
from src.preprocessing import preprocessor
from src.preprocessing import ImageSum
from src.postprocessing.postprocessing import postProcess
from src.Classifiers.Classifier import Classifier
from src.Classifiers.FCN import FCN_Classifier
from src.Classifiers.UNet_Classifier import UNET_Classifier

description = 'CSCI 8630 Project 3 by Team Ball.  ' \
              'This program contains segmentation models to segment neurons from a nouron firing contest at :' \
              'http://neurofinder.codeneuro.org/' \
                'to run kthe program either --train or an exportPath should be supplied. is oexport Path is supplied it has to contain a trained model coresponding to the same classifier selected'

parser = argparse.ArgumentParser(description=description, add_help='How to use', prog='python main.py <options>')

parser.add_argument("-d", "--dataset", default="data/tarin/",
                    help='Path to the training data [DEFAULT: "data/tarin/"]')

parser.add_argument("-ts", "--testset", default=None,
                    help='Path to the testing data [DEFAULT: None]')

parser.add_argument("-m", "--model", default="FCN",
                    help='model to be used in the segmentation can be UNET/FCN/NMF [DEFAULT: "FCN"]')

parser.add_argument("-t", "--train", action="store_true",
                    help='To ensure a model is being trained')

parser.add_argument("-p", "--predict", action="store_true",
                    help='To ensure a segmentation is performed on the test set (This requires --testset to have value)')

parser.add_argument("-sm", "--savemodel", action="store_true",
                    help='Save trained model to a file')

parser.add_argument("-mf", "--modelfile", default=None,
                    help='Path where model should be saved (this is a directory, models can have multiple files)')

parser.add_argument("-ll", "--loglevel", default="info",
                    help='Sets the level for logs that should appear in the log file ( debug, info, warn, error)[DEFAULT : Info]')


parser.add_argument("-pp", "--preprocessor", default="sum",
                    help='Chooses the Preprcessor to be applied ')

parser.add_argument("-ep", "--exportpath", default="sum",
                    help='Chooses the Preprcessor to be applied ')

parser.add_argument("-lf", "--logfile", default="log.log",
                    help="Path to the log file, this file will contain the log records")

# compile arguments
args = parser.parse_args()

# setup logging
logging.basicConfig(filename=args.logfile, level=logging.INFO, filemode="w",
                    format=" %(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s ")

the_preprocessor = None
the_Classifier = None

# set the preprocessor
if (args.preprocessor == "sum"):
    the_preprocessor = ImageSum.ImageSum(images_size=[512, 512], trainingPath=args.dataset, testPath=args.testset,
                                         exportPath=args.exportpath, importPath=args.exportpath )
else:
    the_preprocessor = preprocessor.preprocessor()

# set the set classifier :
if (args.model == "FCN"):
    the_Classifier = FCN_Classifier ( )
elif args.model == "UNET":
    the_Classifier = UNET_Classifier()
else:
    the_Classifier = Classifier()

# -------------- Loading the data

# try to load pre calculated data :
try:

    x_train, y_train, x_test = the_preprocessor.load_from_files()

except FileNotFoundError:
    # if there is no file to load set them as null, they will be loaded autiomatically
    x_train, y_train, x_test = None, None, None

# check if there is no data, read them from input ( this will take time! )
if  ( x_train is None):
    logging.info("Loading data from original data")
    x_train, y_train, x_test = the_preprocessor.preprocess()
    logging.info("Done loading data from original data")
else:
    logging.info("data loaded from pre-calculated files")
# --------------- Loading the data


# --------------- train model!
if( args.train ):
    logging.info("Starting training")
    model = the_Classifier.train(x_train=x_train, y_train=y_train , epochs=1)
    the_Classifier.saveModel( args.exportpath )
    logging.info("Done with training")
else :
    model = the_Classifier.load_model( args.exportpath )

# --------------- train model!

#------------ predict
if( args.predict  and x_test ):
    # run the prediction
    predicted = the_Classifier.predict( x_test )

    # save the results
    postProcess(theDic=predicted,output_file_name="test.json")
# ------------ predict

#
#
#
#
#
#
#
#
#
# filenames = {
#     "00.00.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/0.png",
#     "00.01.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/1.png",
#     "01.00.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/2.png",
#     "01.01.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/3.png",
#     "02.00.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/4.png",
#     "02.01.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/5.png",
#     "03.00.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/6.png",
#     "04.00.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/7.png",
#     "04.01.test": "/media/omido/data2/8360/TeamBall/team-ball/output/output-unet-batch8-epoch1024/8.png",
#
# }
#
# postProcess(fileNames=filenames, output_file_name="output_unet.json")
#

