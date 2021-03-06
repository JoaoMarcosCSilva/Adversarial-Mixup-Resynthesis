from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, MaxPooling2D, UpSampling2D
from lib import losses
from lib.SpectralNormalizationKeras import SpectralNormalization as SN

def get_Encoder(Layers, Hidden_Channels, Starting_Channels, activation = 'relu', input_shape = (64,64,3)):
    inputs = Input(shape = input_shape)
    x = inputs

    channels = Starting_Channels
    
    for l in range(Layers-1):
        x = Conv2D(channels, 3, activation = activation, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(channels, 3, activation = activation, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        channels = int(channels / 2)

    x = Conv2D(Hidden_Channels*2, 3, activation = activation, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(Hidden_Channels, 3, padding = 'same')(x)

    Encoder = keras.Model(inputs, x)

    return Encoder

def get_Decoder(Layers, Hidden_Shape, Encoder_Starting_Channels, instance_norm = False, activation = 'relu', input_shape = (64,64,3)):
    if instance_norm:
        try:
            import tensorflow_addons as tfa
        except:
            print('tensorflow_addons is not available. Instance Normalization will be replaced by Batch Normalization')
    inputs = Input(shape = (Hidden_Shape))
    x = inputs
    
    x = Conv2D(Hidden_Shape[-1]*2, 3, activation = activation, padding = 'same')(x)
    if instance_norm:
        try:
            x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
        except:
            x = BatchNormalization()(x)
    else:
        x = BatchNormalization()(x)
    
    channels = int(Encoder_Starting_Channels / (2**(Layers-1)))

    for l in range(Layers-1):
        channels = channels * 2

        x = UpSampling2D()(x)
        x = Conv2D(channels, 3, activation = activation, padding = 'same')(x)
        if instance_norm:
            try:
                x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
            except:
                x = BatchNormalization()(x)
        else:
            x = BatchNormalization()(x)
        x = Conv2D(channels, 3, activation = activation, padding = 'same')(x)
        if instance_norm:
            try:
                x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
            except:
                x = BatchNormalization()(x)
        else:
            x = BatchNormalization()(x)

    x = Conv2D(3, 3, activation = 'sigmoid', padding = 'same') (x)

    Decoder = keras.Model(inputs, x)

    return Decoder
def get_Model (Layers, Hidden_Channels, Starting_Channels, input_shape = (64,64,3), instance_norm = False, activation = 'relu'):
    inputs = Input(shape = input_shape)
    Encoder = get_Encoder(Layers, Hidden_Channels, Starting_Channels, input_shape = input_shape)
    Decoder = get_Decoder(Layers, Encoder.output_shape[1:], Starting_Channels, instance_norm = instance_norm, input_shape = input_shape)

    x = inputs
    x = Encoder(x)
    x = Decoder(x)

    Model = keras.Model(inputs, x)
    return Encoder, Decoder, Model
    
def get_Discriminator(Layers, Starting_Channels, spectral_norm = False, activation = 'relu', input_shape = (64,64,3), apply_sigmoid = True):
    inputs = Input(shape = input_shape)
    x = inputs
    for l in range(Layers-2):
        if spectral_norm:
            x = SN(Conv2D(Starting_Channels*(2**l), 3, activation = activation, padding = 'same'))(x)
        else:
            x = Conv2D(Starting_Channels*(2**l), 3, activation = activation, padding = 'same')(x)
            x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(10,activation = activation)(x)
    if apply_sigmoid:
        x = Dense(1, activation = 'sigmoid')(x)
    else:
        x = Dense(1)(x)
    Discriminator = keras.Model(inputs,x)
    return Discriminator
    
