import tensorflow as tf
import wandb

class Autoencoder():
    def __init__(self, autoencoder_object, discriminator_object, Lambda):
        self.AE = autoencoder_object
        self.Disc = discriminator_object
        self.Lambda = Lambda

    def train_step_AE(self, batch):
        with tf.GradientTape() as tape:
            x_true = batch
            x_pred = self.AE.autoencode(batch)
            disc_pred = self.Disc.discriminate(x_pred)
            loss_reconstruction = self.AE.loss(batch, x_pred) 
            loss_discrimination = self.Disc.loss(tf.ones(tf.shape(disc_pred)), disc_pred)
            loss = self.Lambda*loss_reconstruction + loss_discrimination
            
        gradients = tape.gradient(loss, self.AE.Model.trainable_variables)
        self.AE.Optimizer.apply_gradients(zip(gradients, self.AE.Model.trainable_variables))
        return loss, gradients, loss_reconstruction, loss_discrimination

    def train_step_Disc(self, batch):
        with tf.GradientTape() as tape:
            disc_pred_real = self.Disc.discriminate(batch)
            disc_pred_fake = self.Disc.discriminate(self.AE.autoencode(batch))
            
            loss_real = self.Disc.loss(tf.ones(tf.shape(disc_pred_real)), disc_pred_real)
            loss_fake = self.Disc.loss(tf.zeros(tf.shape(disc_pred_fake)), disc_pred_fake)

            loss = loss_real + loss_fake
        gradients = tape.gradient(loss, self.Disc.Discriminator.trainable_variables)
        self.Disc.Optimizer.apply_gradients(zip(gradients, self.Disc.Discriminator.trainable_variables))
        return loss, gradients, loss_real, loss_fake
    
    def train(self, epochs, dataset, verbose = True, wandb_run = False):
        for epoch in range(epochs):
            if verbose:
                print('Epoch:',epoch+1)
            for batch in dataset:
                loss_ae, gradients_ae, loss_reconstruction_ae, loss_discrimination_ae = self.train_step_AE(batch)
                loss_disc, gradients_disc, loss_real_disc, loss_fake_disc = self.train_step_Disc(batch)
                if wandb_run:
                    wandb.log({'Epoch': epoch}, commit = False)
                    wandb.log({'Autoencoder Loss': loss_ae, 
                        'Autoencoder Mean Gradient': gradients_ae, 
                        'Autoencoder Reconstruction Loss': loss_reconstruction_ae,
                        'Autoencoder Discrimination Loss': loss_discrimination_ae}, commit = False)
                    wandb.log({'Discriminator Loss': loss_disc, 
                        'Discriminator Mean Gradient': gradients_disc, 
                        'Autoencoder Reconstruction Loss': loss_real_disc,
                        'Autoencoder Discrimination Loss': loss_fake_disc})
                if verbose == 1:
                    ...