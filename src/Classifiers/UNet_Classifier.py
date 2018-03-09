import logging
import numpy as np
import os
import tensorflow as tf
from keras.models import Model

from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, Dropout, Conv2DTranspose, UpSampling2D, Lambda
from keras.layers.normalization import BatchNormalization as bn
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import add
import numpy as np
from keras.regularizers import l2
import cv2
import glob
import h5py
from keras.models import load_model
import os

from src.Classifiers.Classifier import  Classifier

class UNET_Classifier(Classifier):
    """
    This class provides the implementation for the UNet classifier
    """

    def __init__(self, loss_function="dice_coef"):
        self.trained_model = None
        self.classifier_name = "UNET"
        self.data = None

        if (loss_function == "dice_coef"):
            self.metrics_function = UNET_Classifier.dice_coef
            self.loss_function = UNET_Classifier.dice_coef_loss
        elif loss_function == "dice_coef2":
            self.metrics_function = UNET_Classifier.dice_coef2
            self.loss_function = UNET_Classifier.dice_coef_loss2

    def dice_coef2(y_true, y_pred):
        """
        this is a modified version of dice score,
        :param y_true: ground truth
        :param y_pred: predicted
        :return: dice score calculated betweenthe actual and predicted versions
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        U = K.sum(y_true_f * y_pred_f)

        return 1 - intersection / U

    def dice_coef(y_true, y_pred):
        """
        This is dice score implemetation
        :param y_true: ground truth
        :param y_pred: predicted
        :return: dice score calculated betweenthe actual and predicted versions
        """
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        print(K.max(y_true))

        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return - UNET_Classifier.dice_coef(y_true, y_pred)

    def dice_coef_loss2(y_true, y_pred):
        return - UNET_Classifier.dice_coef(y_true, y_pred)

    def load_model(self, import_path):
        """
        overrides the load method to add the costum object
        :param import_path: directory from which model has to be loaded 

        """
        self.trained_model = load_model(os.path.join(import_path, self.classifier_name + ".h5"),
                                        custom_objects={
                                                        'dice_coef_loss': UNET_Classifier.dice_coef_loss,
                                                        'dice_coef': UNET_Classifier.dice_coef})
        logging.info("Loaded Model at : " + os.path.join(import_path, self.classifier_name + ".h5"))

    def train(self, x_train, y_train, epochs=1200, batch_size=4):
        """
        this is the training function for
        :param x_train:
        :param y_train:
        :return:
        """
        l2_lambda = 0.0002
        DropP = 0.3
        kernel_size = 3
        input_shape = (512, 512, 1)
        inputs = Input(input_shape)
        input_prob = Input(input_shape)
        input_prob_inverse = Input(input_shape)
        # Conv3D(filters,(3,3,3),sjfsjf)
        conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
        conv1 = bn()(conv1)
        conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(conv1)
        conv1 = bn()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(DropP)(pool1)

        conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(pool1)
        conv2 = bn()(conv2)
        conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(conv2)
        conv2 = bn()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(DropP)(pool2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            pool2)
        conv3 = bn()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            conv3)
        conv3 = bn()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(DropP)(pool3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            pool3)
        conv4 = bn()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            conv4)
        conv4 = bn()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = Dropout(DropP)(pool4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            pool4)
        conv5 = bn()(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            conv5)
        conv5 = bn()(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], name='up6',
                          axis=3)
        up6 = Dropout(DropP)(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            up6)
        conv6 = bn()(conv6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            conv6)

        conv6 = bn()(conv6)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], name='up7',
                          axis=3)
        up7 = Dropout(DropP)(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            up7)
        conv7 = bn()(conv7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(
            conv7)
        conv7 = bn()(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], name='up8',
                          axis=3)
        up8 = Dropout(DropP)(up8)
        conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(up8)
        conv8 = bn()(conv8)
        conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(conv8)
        conv8 = bn()(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], name='up9',
                          axis=3)
        up9 = Dropout(DropP)(up9)
        conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(up9)
        conv9 = bn()(conv9)
        conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda))(conv9)
        conv9 = bn()(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv10')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        model.compile(optimizer=Adam(lr=1e-5), loss=UNET_Classifier.dice_coef_loss, metrics=[UNET_Classifier.dice_coef])
        print(model.summary())

        # training network
        model.fit([x_train], [y_train], batch_size=batch_size, epochs=epochs, shuffle=True)

        # set as class's model to be used for prediction
        self.trained_model = model

        return model
