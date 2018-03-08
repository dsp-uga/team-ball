""""This file manages the postprocessing of the objects in the images"""

import numpy as np
import cv2
import json

def postProcess ( fileNames =None , theDic=None, output_file_name = "output.json" ):
    """
    this function handels the postprocessing of values,
    :param fileNames: if out put is written to files and funciton is supposed to load them from files this dictionary contains their names in format {"sampleName":"filename"}
    :param theDic: if output is to be loaded from a dictionary, this is the dictionary in format { "sampleName" : ValueArray }
    :param output_file_name: name of the file to which the json results should be written
    """

    # validate input:
    if not( fileNames  or theDic ):
        raise ValueError('One of the values filename or theDic has to be set!')


    file_name_values_dic ={}

    if( fileNames ):
        # if images should be loaded from files do so!
        for file_name in fileNames:
            file_name_values_dic[ file_name ] = cv2.imread( fileNames[ file_name], 0)
    else :

        file_name_values_dic = theDic

    final_dic = []

    for key in file_name_values_dic :

        # find connected components
        x, markers = cv2.connectedComponents( file_name_values_dic[key] )

        #convert them to the writeable format
        temp =   []
        for i in range (1,x) :
            temp.append( list(  [int(pick[0]),int(pick[1])]  for pick in np.argwhere(markers==i)  ))

        # convert lists to dictionaries
        temp = [ {"coordinates": pick } for pick in temp ]

        # add sample name info to the dictionary
        temp_dic = { "dataset": key,
                     "regions": temp
                     }

        # add to the completed object
        final_dic.append( temp_dic )

    # create json file string
    json_dump = json.dumps( final_dic,  indent=4 )
    # save the json file string
    with open( output_file_name,'w' ) as file:
        file.write(json_dump)

