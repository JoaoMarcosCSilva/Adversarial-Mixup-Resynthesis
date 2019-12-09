import tensorflow as tf
import numpy as np
from tensorflow import keras

from lib import losses
class Autoencoder():
    def __init__(self, Encoder, Decoder, Discriminator = None, Autoencoder_Optimizer = None, Discriminator_Optimizer = None):
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Discriminator = Discriminator

        inputs = keras.layers.Input(shape = Encoder.input_shape[1:])
        self.Autoencoder = keras.Model(inputs, Decoder(Encoder(inputs)))

        self.Autoencoder_Optimizer = Autoencoder_Optimizer if Autoencoder_Optimizer != None else keras.optimizers.Adam()
        self.Discriminator_Optimizer = Discriminator_Optimizer if Discriminator_Optimizer != None else keras.optimizers.Adam()

    def autoencoder_loss(self, x_true, x_pred):
        return losses.reconstruction_loss(x_true, x_pred)
    
    def discriminator_loss(self, y_true, y_pred):
        return losses.discriminator_loss(y_true, y_pred)
    
    def evaluate(self, dataset):
        l = 0
        for step, batch in enumerate(dataset):
            l += self.loss(batch, autoencode(batch))
        return l/(step+1)
    
    @tf.function
    def autoencoder_train_step(self, batch):
        with tf.GradientTape() as tape:
            pred = self.Autoencoder(batch)
            loss = self.autoencoder_loss(batch, pred)
        gradients = tape.gradient(loss, self.Autoencoder.trainable_variables)
        self.Autoencoder_Optimizer.apply_gradients(zip(gradients, self.Autoencoder.trainable_variables))
        return loss, gradients
    
    @tf.function
    def discriminator_train_step(self, batch):
        # The base autoencoder's does not have a discriminator, so this function does nothing
        return 
    
    def wandb_step(self, metrics_dict, epoch = None):
        if epoch == None:
            wandb.log(metrics_dict)
        else:
            wandb.log({**{'Epoch':epoch+1}, **metrics_dict})
    
    def get_progress_bar(self, dataset):
        return keras.utils.Progbar(int(tf.data.experimental.cardinality(dataset)) - 1)

    def progress_bar_step(self, progress_bar, step, dict, subset = None):
        if type(subset) is not list:
            progress_bar.update(step, list(zip(dict.keys(), dict.values())))
        else:
            self.progress_bar_step(progress_bar, step, {k:dict[k] for k in dict if k in subset})
            
    
    def train(self, epochs, dataset, verbose = 1, log_wandb = 0):
        if log_wandb:
            import wandb
        for epoch in range(epochs):
            if verbose:
                print('Epoch:',epoch+1)
                progress_bar = self.get_progress_bar(dataset)
                
            for step, batch in enumerate(dataset):
                p = self.Autoencoder(batch)
                autoencoder_loss, autoencoder_gradients = self.autoencoder_train_step(batch)
                
                metrics_dict = {'Autoencoder Reconstruction Loss':autoencoder_loss.numpy(),
                                'Autoencoder Mean Gradient':np.mean([np.mean(i.numpy()) for i in autoencoder_gradients])}
                if log_wandb:
                    wandb_step(metrics_dict, epoch)
                
                if verbose:
                    self.progress_bar_step(progress_bar, step, metrics_dict)
                    