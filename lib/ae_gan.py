import tensorflow as tf
import numpy as np

from lib import baseline

class Autoencoder(baseline.Autoencoder):
    def __init__(self, Encoder, Decoder, Discriminator, Lambda, Autoencoder_Optimizer = None, Discriminator_Optimizer = None):
        super().__init__(Encoder, Decoder, Discriminator, Autoencoder_Optimizer, Discriminator_Optimizer)
        self.Lambda = Lambda

    @tf.function
    def autoencoder_train_step(self, batch):
        with tf.GradientTape() as tape:
            x_true = batch
            x_pred = self.Autoencoder(batch)
            disc_pred = self.Discriminator(x_pred)
            loss_reconstruction = self.autoencoder_loss(batch, x_pred) 
            loss_discrimination = self.discriminator_loss(tf.ones(tf.shape(disc_pred)), disc_pred)
            loss = self.Lambda*loss_reconstruction + loss_discrimination
            
        gradients = tape.gradient(loss, self.Autoencoder.trainable_variables)
        self.Autoencoder_Optimizer.apply_gradients(zip(gradients, self.Autoencoder.trainable_variables))
        return loss, gradients, loss_reconstruction, loss_discrimination

    @tf.function
    def discriminator_train_step(self, batch):
        with tf.GradientTape() as tape:
            disc_pred_real = self.Discriminator(batch)
            disc_pred_fake = self.Discriminator(self.Autoencoder(batch))
            
            loss_real = self.discriminator_loss(tf.ones(tf.shape(disc_pred_real)), disc_pred_real)
            loss_fake = self.discriminator_loss(tf.zeros(tf.shape(disc_pred_fake)), disc_pred_fake)

            loss = loss_real + loss_fake
        gradients = tape.gradient(loss, self.Discriminator.trainable_variables)
        self.Discriminator_Optimizer.apply_gradients(zip(gradients, self.Discriminator.trainable_variables))
        return loss, gradients, loss_real, loss_fake
    
    def train(self, epochs, dataset, verbose = 1, log_wandb = 0, wandb_every = 1, disc_every = 1):
        if log_wandb:
            import wandb
        for epoch in range(epochs):
            if verbose:
                print('Epoch:',epoch+1)
                progress_bar = self.get_progress_bar(dataset)
            for step, batch in enumerate(dataset):
                loss_ae, gradients_ae, loss_reconstruction_ae, loss_discrimination_ae = self.autoencoder_train_step(batch)
                
                if step % disc_every == 0:
                    loss_disc, gradients_disc, loss_real_disc, loss_fake_disc = self.discriminator_train_step(batch)
                
                metrics_dict = {'Autoencoder Loss': loss_ae.numpy(), 
                            'Autoencoder Mean Gradient': np.mean([np.mean(i.numpy()) for i in gradients_ae]), 
                            'Autoencoder Reconstruction Loss': loss_reconstruction_ae.numpy(),
                            'Autoencoder Discrimination Loss': loss_discrimination_ae.numpy(),
                            'Discriminator Loss': loss_disc.numpy(), 
                            'Discriminator Mean Gradient': np.mean([np.mean(i.numpy()) for i in gradients_disc]),
                            'Discriminator Real Loss': loss_real_disc.numpy(),
                            'Discriminator Fake Loss': loss_fake_disc.numpy()}

                if log_wandb:
                    if step % wandb_every == 0:
                        wandb_step(metrics_dict, epoch)

                if verbose == 1:
                    self.progress_bar_step(progress_bar, step, metrics_dict, ['Autoencoder Loss','Discriminator Loss', 'Autoencoder Reconstruction Loss'])
                if verbose == 2:
                    self.progress_bar_step(progress_bar, step, metrics_dict)
                