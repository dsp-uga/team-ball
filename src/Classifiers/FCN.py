"""
this class is the FCN ( Fully Convelutional Network ) implementation as a classifier
"""

from src.Classifiers.Classifier import Classifier
from keras.models import Model
from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, Dropout, Conv2DTranspose, UpSampling2D, Lambda
from keras.optimizers import Adam
from keras.layers.merge import add
import numpy as np
from keras.regularizers import l2
import keras.backend as K
import keras_fcn.backend as K1
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.engine import InputSpec
import logging
from keras.models import load_model
import os


class BilinearUpSampling2D(Layer):
    """Upsampling2D with bilinear interpolation."""

    def __init__(self, target_shape=None, data_format=None, **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {
            'channels_last', 'channels_first'}
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0],
                    self.target_size[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1],
                    self.target_size[0], self.target_size[1])

    def call(self, inputs):
        return K1.resize_images(inputs, size=self.target_size,
                                method='bilinear')

    def get_config(self):
        config = {'target_shape': self.target_shape,
                'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CroppingLike2D(Layer):
    def __init__(self, target_shape, offset=None, data_format=None,
                 **kwargs):
        """Crop to target.
        If only one `offset` is set, then all dimensions are offset by this amount.
        """
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_shape = target_shape
        if offset is None or offset == 'centered':
            self.offset = 'centered'
        elif isinstance(offset, int):
            self.offset = (offset, offset)
        elif hasattr(offset, '__len__'):
            if len(offset) != 2:
                raise ValueError('`offset` should have two elements. '
                                 'Found: ' + str(offset))
            self.offset = offset
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    self.target_shape[2],
                    self.target_shape[3])
        else:
            return (input_shape[0],
                    self.target_shape[1],
                    self.target_shape[2],
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        if self.data_format == 'channels_first':
            input_height = input_shape[2]
            input_width = input_shape[3]
            target_height = self.target_shape[2]
            target_width = self.target_shape[3]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))

            return inputs[:,
                          :,
                          self.offset[0]:self.offset[0] + target_height,
                          self.offset[1]:self.offset[1] + target_width]
        elif self.data_format == 'channels_last':
            input_height = input_shape[1]
            input_width = input_shape[2]
            target_height = self.target_shape[1]
            target_width = self.target_shape[2]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))
            output = inputs[:,
                            self.offset[0]:self.offset[0] + target_height,
                            self.offset[1]:self.offset[1] + target_width,
                            :]
            return output

    def get_config(self):
        config = {'target_shape': self.target_shape,
                  'offset': self.offset,
                  'data_format': self.data_format}
        base_config = super(CroppingLike2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class FCN_Classifier( Classifier ):
    """
    This class provides the implementation for the FCN classifier
    """
    def __init__(self , loss_function = "dice_coef" ):
        self.trained_model = None
        self.classifier_name = "FCN"
        self.data = None

        if( loss_function =="dice_coef" ):
            self.metrics_function = FCN_Classifier.dice_coef
            self.loss_function = FCN_Classifier.dice_coef_loss
        elif loss_function =="dice_coef2":
            self.metrics_function = FCN_Classifier.dice_coef2
            self.loss_function = FCN_Classifier.dice_coef_loss2


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
        return - FCN_Classifier.dice_coef(y_true, y_pred)
    def dice_coef_loss2(y_true, y_pred):
        return - FCN_Classifier.dice_coef(y_true, y_pred)

    def load_model(self , import_path):
        """
        overrides the load method to add the costum object
        :param import_path: directory from which model has to be loaded 
    
        """
        self.trained_model =load_model(os.path.join(import_path, self.classifier_name + ".h5"), custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D,
                                                                                                                'dice_coef_loss': FCN_Classifier.dice_coef_loss,
                                                                                                                'dice_coef': FCN_Classifier.dice_coef})
        logging.info("Loaded Model at : " + os.path.join(import_path, self.classifier_name + ".h5"))



    def train (self, x_train, y_train , epochs= 1200 , batch_size=4):
        """
        this is the training function for
        :param x_train:
        :param y_train:
        :return:
        """
        weight_decay = 0

        x_train = x_train.reshape(x_train.shape + (1,))
        y_train = y_train.reshape(y_train.shape + (1,))


        # Block 1
        img_input = Input(shape=(512, 512, 1))
        block1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
                              kernel_regularizer=l2(weight_decay))(img_input)
        block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',
                              kernel_regularizer=l2(weight_decay))(block1_conv1)
        block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

        # Block 2
        block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
                              kernel_regularizer=l2(weight_decay))(block1_pool)
        block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
                              kernel_regularizer=l2(weight_decay))(block2_conv1)
        block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

        # Block 3
        block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
                              kernel_regularizer=l2(weight_decay))(block2_pool)
        block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',
                              kernel_regularizer=l2(weight_decay))(block3_conv1)
        block3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
                              kernel_regularizer=l2(weight_decay))(block3_conv2)
        block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv3)

        # Block 4
        block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
                              kernel_regularizer=l2(weight_decay))(block3_pool)
        block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
                              kernel_regularizer=l2(weight_decay))(block4_conv1)
        block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
                              kernel_regularizer=l2(weight_decay))(block4_conv2)
        block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)

        # Block 5
        block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
                              kernel_regularizer=l2(weight_decay))(block4_pool)
        block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
                              kernel_regularizer=l2(weight_decay))(block5_conv1)
        block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
                              kernel_regularizer=l2(weight_decay))(block5_conv2)
        block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_conv3)

        block5_fc6 = Conv2D(4096, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal',
                            name='block5_fc6', kernel_regularizer=l2(weight_decay))(block5_pool)
        dropout_1 = Dropout(0.5)(block5_fc6)
        blockk5_fc7 = Conv2D(4096, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                             name='blockk5_fc7', kernel_regularizer=l2(weight_decay))(dropout_1)
        dropout_2 = Dropout(0.5)(blockk5_fc7)

        score_feat1 = Conv2D(1, (3, 3), activation='linear', padding='same', name='score_feat1',
                             kernel_regularizer=l2(weight_decay))(dropout_2)
        score_feat2 = Conv2D(1, (3, 3), activation='linear', padding='same', name='score_feat2',
                             kernel_regularizer=l2(weight_decay))(block4_pool)
        score_feat3 = Conv2D(1, (3, 3), activation='linear', padding='same', name='score_feat3',
                             kernel_regularizer=l2(weight_decay))(block3_pool)

        # scale_feat2 = Lambda(scaling, arguments={'ss':1},name='scale_feat3')(score_feat2)
        # scale_feat3 = Lambda(scaling, arguments={'ss': scale},name='scale_feat3')(score_feat3)

        scale_feat2 = Lambda(lambda x: x * 2, name='scale_feat2')(score_feat2)
        scale_feat3 = Lambda(lambda x: x * 2, name='scale_feat3')(score_feat3)

        upscore_feat1 = BilinearUpSampling2D(target_shape=(None, 32, 32, None), name='upscore_feat1')(score_feat1)

        add_1 = add([upscore_feat1, scale_feat2])
        upscore_feat2 = BilinearUpSampling2D(target_shape=(None, 64, 64, None), name='upscore_feat2')(add_1)

        add_2 = add([upscore_feat2, scale_feat3])
        upscore_feat3 = BilinearUpSampling2D(target_shape=(None, 512, 512, None), name='upscore_feat3')(add_2)

        output = Activation('sigmoid')(upscore_feat3)

        model = Model(inputs=[img_input], outputs=output)

        model.compile(optimizer=Adam(lr=1e-5), loss=self.loss_function, metrics=[self.metrics_function])
        print(model.summary())

        # training network
        model.fit([x_train], [y_train], batch_size=batch_size, epochs=epochs, shuffle=True)

        # set as class's model to be used for prediction
        self.trained_model = model

        return model
