
def UNet(input_shape,learn_rate=1e-3):
    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape)
    
    # Block1 
    conv1_1 = Conv2D( 64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    conv1_2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv1_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    drop_1 = Dropout(DropP)(pool_1)


    # Block2
    conv2_1 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_1)
    conv2_2 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv2_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    drop_2 = Dropout(DropP)(pool_2)


    # Block3
    conv3_1 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_2)
    conv3_2 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    drop_3 = Dropout(DropP)(pool_3)


    # Block4
    conv4_1 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_3)
    conv4_2 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv4_1)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    drop_4 = Dropout(DropP)(pool_4)


    # Block5
    conv5_1 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_4)
    conv5_2 = Conv2D(1024, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv5_1)
    
    # Block6
    up6 = concatenate([Conv2DTranspose(512,(2, 2), strides=(2, 2), padding='same')(conv5_2), conv4_2],name='up6', axis=3)
    drop_6= Dropout(DropP)(up6)
    conv6_1 = Conv2D(512,(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_6)
    conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv6_1)

    # Block7               
    up7 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv6_2), conv3_2],name='up7', axis=3)
    drop_7 = Dropout(DropP)(up7)
    conv7_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_7)
    conv7_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv7_1)

    # Block8 
    up8 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv7_2), conv2_2],name='up8', axis=3)
    drop_8 = Dropout(DropP)(up8)
    conv8_1 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_8)
    conv8_2 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv8_1)


    # Block9
    up9 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv8_2), conv1_2],name='up9',axis=3)
    drop_9 = Dropout(DropP)(up9)
    conv9_1 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(drop_9)
    conv9_2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv9_1)
    conv9_3 = Conv2D(2, (1, 1), activation='sigmoid',name='conv9_3')(conv9_2)
    

    model = Model(inputs=[inputs], outputs=[conv9_3])
    model.compile(optimizer=Adam(lr=1e-5), loss={'conv10': neg_dice_coef_loss, 'xxconv10': 'mse',})
    return model


model=UNet(input_shape=(256,256,1))
print(model.summary())

    