import tensorflow as tf
import numpy as np
import wandb

class Base():
    def __init__(self, Encoder, Decoder, Discriminator = None, Autoencoder_Optimizer = None, Discriminator_Optimizer = None):
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Discriminator = Discriminator
        self.Autoencoder = keras.Sequential(Encoder, Decoder)
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
            loss = self.AE.loss(batch, self.AE.autoencode(batch))
        gradients = tape.gradient(loss, self.AE.Model.trainable_variables)
        self.AE.Optimizer.apply_gradients(zip(gradients, self.AE.Model.trainable_variables))
        return loss, gradients
    
    @tf.function
    def discriminator_train_step(self, batch):
        # The base autoencoder's does not have a discriminator, so this function does nothing
        return 
    
    def wandb_step(self, dict):
        wandb.log(dict)
    
    def progress_bar_step(self, progress_bar, step, dict):
        progress_bar.update(step, list(zip(dict.keys(), dict.values())))
    
    def train(self, epochs, dataset, verbose = 1, log_wandb = 0):
        for epoch in range(epochs):
            if verbose:
                print('Epoch:',epoch+1)
                progress_bar = keras.utils.Progbar()
                
            for step, batch in enumerate(dataset):
                autoencoder_loss, autoencoder_gradients = autoencoder_train_step(batch)
                
                metrics_dict = {'Autoencoder Reconstruction Loss':autoencoder_loss.numpy(),
                                'Autoencoder Mean Gradient':np.mean([np.mean(i.numpy()) for i in gradients])}
                if log_wandb:
                    wandb_step({**{'Epoch':epoch+1}, **metrics_dict})
                
                if verbose:
                    progress_bar_step(progress_bar, step, metrics_dict)
                    