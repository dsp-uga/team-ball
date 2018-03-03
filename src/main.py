"""
this file contains the main runner for the project
"""

import argparse
import sys
import os
import logging


description = 'CSCI 8630 Project 3 by Team Ball.  ' \
              'This program contains segmentation models to segment neurons from a nouron firing contest at :'\
    'http://neurofinder.codeneuro.org/'

parser = argparse.ArgumentParser(description=description, add_help='How to use', prog='python main.py <options>')


parser.add_argument("-d", "--dataset", default="data/tarin/",
    help='Path to the training data [DEFAULT: "data/tarin/"]')

parser.add_argument("-t", "--testset", default=None,
    help='Path to the testing data [DEFAULT: None]')

parser.add_argument("-m", "--model", default="FCN",
    help='model to be used in the segmentation [DEFAULT: "FCN"]')

parser.add_argument("-t", "--train", action="store_true",
    help='To ensure a model is being trained')

parser.add_argument("-p", "--predict", action="store_true",
    help='To ensure a segmentation is performed on the test set (This requires --testset to have value)')

parser.add_argument("-sm", "--savemodel", action="store_true",
    help='Save trained model to a file' )

parser.add_argument("-mf", "--modelfile", default=None,
    help='Path where model should be saved (this is a directory, models can have multiple files)')

parser.add_argument("-ll", "--loglevel", default="info",
    help='Sets the level for logs that should appear in the log file ( debug, info, warn, error)[DEFAULT : Info]')

parser.add_argument("-tf", "--trainingfile", default=None,
    help='Path where training file should be loaded/saved to')

parser.add_argument("-lf", "--logfile", default="log.log", help="Path to the log file, this file will contain the log records")

# compile arguments
args = parser.parse_args()

# setup logging
logging.basicConfig( filename=args.logfile , level=logging.INFO, filemode="w", format=" %(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s "  )


# TODO : add file loading

# TODO : add preprocessing

# TODO : add training ( optional )

# TODO : add prediction

# TODO : add postprocesing



