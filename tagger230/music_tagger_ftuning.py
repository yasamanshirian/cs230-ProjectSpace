from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from tagger_net import MusicTaggerCRNN,pop_layer
K.set_image_dim_ordering('th')


def music_tagger_wrapper(load_weights):
    
    '''  

    This version is going to be a fine-tuned version, pre-trained on:
    (https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning)
    
    '''    
    
    
    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2
    # Initialize model
    model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))



    if load_weights:
        model.load_weights('pre_trained/weights/pre_trained_crnn_net_gru_adam_epoch_40.h5', by_name=True)


    #popping 2 GRU layers + softmax layer to add 2 extra Convd layers 
    # pop_layer(model)
    # pop_layer(model)
    # pop_layer(model)
    # pop_layer(model)

    model = Model(model.input, model.output)
    # model.summary()


    last = model.get_layer('final_drop')
    x = last.output
    output = Dense(7, activation='sigmoid', name='output',trainable=True)(x)
    model = Model(model.input, output)


    #This section need more attention
    #  # Conv block 5
    # x = Convolution2D(128, (3, 3), padding='same', name='conv5', trainable=True)(x)
    # x = BatchNormalization(axis=channel_axis,  name='bn5', trainable=True)(x)
    # x = ELU()(x)
    # x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool5', trainable=True)(x)
    # x = Dropout(0.1, name='dropout5', trainable=True)(x)

    # # Conv block 6
    # x = Convolution2D(128, (3, 3), padding='same', name='conv6', trainable=True)(x)
    # x = BatchNormalization(axis=channel_axis, name='bn6', trainable=True)(x)
    # x = ELU()(x)
    # x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool6', trainable=True)(x)
    # x = Dropout(0.1, name='dropout6', trainable=True)(x)

    # # reshaping
    # if K.image_dim_ordering() == 'th':
    #      x = Permute((3, 1, 2))(x)
    # x = Reshape((15, 128))(x)

    # # GRU block 1, output
    # x = GRU(32, return_sequences=True, name='gru1')(x)
    # x = Dropout(0.3, name='final_drop')(x)

    # #output
    # output = Dense(7, activation='sigmoid', name='output')(x)
    # model = Model(model.input, output)

    return model 

