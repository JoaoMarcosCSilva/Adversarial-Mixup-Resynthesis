from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, MaxPooling2D, UpSampling2D
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
        return self.Autoencoder(x)

    def loss(self,y_true, y_pred):
        return losses.reconstruction_loss(y_true, y_pred)
    def gradients(self,y_true, y_pred):
        with tf.GradientTape() as tape:
            loss = self.loss(y_true, y_pred)
        return loss, tape.gradient(loss, self.Model.trainable_variables)
        
class Discriminator_Object():
    def __init__(self,Layers, Starting_Channels):
        self.Discriminator = get_Discriminator(Layers, Starting_Channels)
    
    def discriminate(self, x):
        return self.Discriminator(x)

    def loss(self, y_true, y_pred):
        return discriminator_loss(y_true, y_pred)
    def gradients(self,y_true, y_pred):
        with tf.GradientTape() as tape:
            loss = self.loss(y_true, y_pred)
        return loss, tape.gradient(loss, self.Discriminator.trainable_variables)

class Autoencoder():
    def __init__(self, autoencoder_object, discriminator_object):
        self.AE = autoencoder_object
        self.Disc = discriminator_object
    def train_step_AE(self, batch):
        loss, gradients = self.AE.gradients(batch, self.AE.autoencode(batch))
        self.AE.Optimizer.apply_gradients(zip(gradients, self.AE.Model.trainable_variables))
    def train_step_Disc(self, batch):
        disc_pred_real = self.Disc.discriminate(batch)
        disc_pred_fake = self.Disc.discriminate(self.AE.autoencode(batch))
        loss_real, gradients_real = self.Disc.gradients(tf.ones(tf.shape(disc_pred_real)), disc_pred_real)
        loss_fake, gradients_fake = self.Disc.gradients(tf.zeros(tf.shape(disc_pred_fake)), disc_pred_fake)
        gradients = gradients_real + gradients_fake
        self.Disc.Optimizer.apply_gradients(zip(gradients, self.Disc.Discriminator.trainable_variables))





def get_Encoder(Layers, Hidden_Channels, Starting_Channels):
    
    inputs = Input(shape = (64,64,3))
    x = inputs

    channels = Starting_Channels
    
    for l in range(Layers-1):
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
        x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(10,activation='relu')(x)
    x = Dense(1, activation = 'sigmoid')(x)
    Discriminator = keras.Model(inputs,x)
    return Discriminator
    