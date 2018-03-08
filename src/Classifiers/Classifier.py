"""
this is a the abstract class for the classifers
"""
import os
import keras
from keras.models import load_model
import logging

class Classifier :

    def __init__(self):
        self.trained_model = None
        self.data = None
        self.classifier_name = None

        return

    def saveModel(self , export_path ):
        """
        this function is in charge of saving the model and it's weights
        :return:
        """
        if( self.trained_model ):
            self.trained_model.save( os.path.join( export_path ,  self.classifier_name+ ".h5"  ) )

        logging.info(  "Saved Model to : "+os.path.join( export_path ,  self.classifier_name+ ".h5" ))
        return

    def load_model(self , import_path):
        """
        loads model from file
        :param import_path:
        :return:  returns the model and sets it as the model in the class as well
        """
        self.trained_model = load_model(  os.path.join( import_path ,  self.classifier_name+ ".h5"  ))
        logging.info("Loaded Model at : " + os.path.join(import_path, self.classifier_name + ".h5"))
        return self.trained_model

    def train(self, x_train , y_train):
        """
        this funciton  trains the model which is defined in it's body and
        saves the model in the class for further prediction
        :return:
        """

        return self.trained_model

    def predict(self, data_dic):
        """
        this function runs the prediction on the data
        :param data_dic:
        :return: the predicted values
        """
        ret = [{x: self.trained_model.predict([data_dic[x]])[0]} for x in data_dic]

        return ret