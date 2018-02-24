
def UNet(input_shape,learn_rate=1e-3):
    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape)
    input_prob=Input(input_shape)
    input_prob_inverse=Input(input_shape)
    
    # Block1 
    conv1_1 = Conv2D( 64, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    conv1_2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv1_1)
    conv1_3 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv1_3)
    
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv1_3)
    drop_1 = Dropout(DropP)(pool_1)





    # Block2
    conv2_1 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool_1)
    conv2_2 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv2_1)
    conv2_3 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv2_2)
    
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv2_3)
    drop_2 = Dropout(DropP)(pool_2)


    # Block3
    conv3_1 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool_2)
    conv3_2 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv3_1)
    conv3_3 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv3_2)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)
    drop_3 = Dropout(DropP)(pool_3)


    # Block4
    conv4_1 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool_3)
    conv4_2 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv4_1)
    conv4_3 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv4_2)
    
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)
    drop_4 = Dropout(DropP)(pool_4)


    # Block5
    conv5 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
    conv5 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv5)
    conv5 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv5)

    conv5 = bn()(conv5)
    
    up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5), conv4],name='up6', axis=3)

    up6 = Dropout(DropP)(up6)


    conv6 = Conv2D(256,(3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
    
    conv6 = bn()(conv6)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv6)

    conv6 = bn()(conv6)

    up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6), conv3],name='up7', axis=3)

    up7 = Dropout(DropP)(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up7)

    conv7 = bn()(conv7)
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv7)

    conv7 = bn()(conv7)

    up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7), conv2],name='up8', axis=3)

    up8 = Dropout(DropP)(up8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up8)

    conv8 = bn()(conv8)

    
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv8)

    conv8 = bn()(conv8)

    up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8), conv1],name='up9',axis=3)

    up9 = Dropout(DropP)(up9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up9)
    
    conv9 = bn()(conv9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv9)
   
    conv9 = bn()(conv9)
    #prob_skull=Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    #prob_skull=keras.layers.multiply([input_prob,prob_skull])
    #prob_skull=keras.layers.add([input_prob_inverse,prob_skull])
    #prob_skull=keras.layers.multiply([input_prob_inverse,prob_skull])
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid',name='conv10')(conv9)
    


    