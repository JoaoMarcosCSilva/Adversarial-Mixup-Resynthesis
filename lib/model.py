from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, MaxPooling2D, UpSampling2D
import tensorflow as tf
from lib import losses

class Autoencoder_Object():
    def __init__(self, Layers, Hidden_Channels, Starting_Channels, lr = 0.001):
        self.Encoder, self.Decoder, self.Model = get_Model(Layers, Hidden_Channels, Starting_Channels)
        self.Optimizer = keras.optimizers.Adam(lr)

    def encode(self,x):
        return self.Encoder(x)
    def decode(self,x):
        return self.Decoder(x)
    def autoencode(self,x):
        return self.Model(x)

    def loss(self,y_true, y_pred):
        return losses.reconstruction_loss(y_true, y_pred)
    
    def evaluate(self, dataset):
        l = 0
        i = 0
        for batch in dataset:
            l += self.loss(batch, autoencode(batch))
            i += 1
        return l/i

class Discriminator_Object():
    def __init__(self,Layers, Starting_Channels, lr = 0.001):
        self.Discriminator = get_Discriminator(Layers, Starting_Channels)
        self.Optimizer = keras.optimizers.Adam(lr)
    def discriminate(self, x):
        return self.Discriminator(x)

    def loss(self, y_true, y_pred):
        return losses.discriminator_loss(y_true, y_pred)

def get_Encoder(Layers, Hidden_Channels, Starting_Channels):
    
    inputs = Input(shape = (64,64,3))
    x = inputs

    channels = Starting_Channels
    
    for l in range(Layers-1):
        x = Conv2D(channels, 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(channels, 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        channels = int(channels / 2)

    x = Conv2D(Hidden_Channels*2, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(Hidden_Channels, 3, padding = 'same')(x)

    Encoder = keras.Model(inputs, x)

    return Encoder
def get_Decoder(Layers, Hidden_Shape, Encoder_Starting_Channels):

    inputs = Input(shape = (Hidden_Shape))
    x = inputs
    
    x = Conv2D(Hidden_Shape[-1]*2, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    
    channels = int(Encoder_Starting_Channels / (2**(Layers-1)))

    for l in range(Layers-1):
        channels = channels * 2

        x = UpSampling2D()(x)
        x = Conv2D(channels, 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(channels, 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)

    x = Conv2D(3, 3, activation = 'sigmoid', padding = 'same') (x)

    Decoder = keras.Model(inputs, x)

    return Decoder
def get_Model (Layers, Hidden_Channels, Starting_Channels):
    inputs = Input(shape = (64,64,3))
    Encoder = get_Encoder(Layers, Hidden_Channels, Starting_Channels)
    Decoder = get_Decoder(Layers, Encoder.output_shape[1:], Starting_Channels)

    x = inputs
    x = Encoder(x)
    x = Decoder(x)

    Model = keras.Model(inputs, x)
    return Encoder, Decoder, Model
def get_Discriminator(Layers, Starting_Channels):
    inputs = Input(shape = (64,64,3))
    x = inputs
    for l in range(Layers-2):
        x = Conv2D(Starting_Channels*(2**l), 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(10,activation='relu')(x)
    x = Dense(1, activation = 'sigmoid')(x)
    Discriminator = keras.Model(inputs,x)
    return Discriminator
    