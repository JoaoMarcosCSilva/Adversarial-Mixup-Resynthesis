import tensorflow as tf
import numpy as np

from lib import baseline,mixup

class Autoencoder(baseline.Autoencoder):
    def __init__(self, Encoder, Decoder, Discriminator, Lambda, Beta = 0.0, Autoencoder_Optimizer = None, Discriminator_Optimizer = None, Mixup = mixup.linear):
        super().__init__(Encoder, Decoder, Discriminator, Autoencoder_Optimizer, Discriminator_Optimizer)
        self.Lambda = Lambda
        self.Beta = Beta
        self.Mixup = Mixup

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
            x_true = batch
            h_pred = self.Encoder(batch)
            x_pred = self.Decoder(h_pred)

            h_mixup = self.Mixup(h_pred)[0]
            x_mixup = self.Decoder(h_mixup)

            h_consistency = self.Encoder(x_mixup)

            disc_true = self.Discriminator(x_true)
            disc_pred = self.Discriminator(x_pred)
            disc_mixup = self.Discriminator(x_mixup)

            ae_reconstruction_loss = self.autoencoder_loss(batch, x_pred)
            ae_discrimination_loss = -1 * tf.reduce_mean(tf.math.log(disc_pred) + 1e-7)
            ae_mixup_loss = -1 * tf.reduce_mean(tf.math.log(disc_mixup) + 1e-7)
            ae_consistency_loss = tf.reduce_mean(tf.keras.losses.mse(h_mixup, h_consistency))

            disc_true_loss = self.discriminator_loss(tf.ones_like(disc_true), disc_true)
            disc_fake_loss = self.discriminator_loss(tf.zeros_like(disc_pred), disc_pred)
            disc_mixup_loss = self.discriminator_loss(tf.zeros_like(disc_mixup), disc_mixup)

            ae_loss = self.Lambda*ae_reconstruction_loss + ae_discrimination_loss + ae_mixup_loss + self.Beta*ae_consistency_loss
            disc_loss = disc_true_loss + disc_fake_loss + disc_mixup_loss
        ae_gradients = ae_tape.gradient(ae_loss, self.Autoencoder.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.Discriminator.trainable_variables)
        self.Autoencoder_Optimizer.apply_gradients(zip(ae_gradients, self.Autoencoder.trainable_variables))
        self.Discriminator_Optimizer.apply_gradients(zip(disc_gradients, self.Discriminator.trainable_variables))

        return ae_loss, ae_gradients, ae_reconstruction_loss, ae_discrimination_loss, ae_mixup_loss, ae_consistency_loss, disc_loss, disc_gradients, disc_true_loss, disc_fake_loss, disc_mixup_loss

    @tf.function
    def autoencoder_train_step(self, batch):
        with tf.GradientTape() as tape:
            x_true = batch
            h_pred = self.Encoder(batch)
            x_pred = self.Decoder(h_pred)

            disc_pred = self.Discriminator(x_pred)
            loss_reconstruction = self.autoencoder_loss(batch, x_pred) 
            loss_discrimination = -1 * tf.reduce_mean(tf.math.log(disc_pred) + 1e-7)
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
    
    def train(self, epochs, dataset, verbose = 1, log_wandb = 0, plot_data = None):
        for epoch in range(epochs):
            if verbose:
                print('Epoch:',epoch+1)
                progress_bar = self.get_progress_bar(dataset)
            for step, batch in enumerate(dataset):
                loss_ae, gradients_ae, loss_reconstruction_ae, loss_discrimination_ae, loss_mixup_ae, loss_consistency_ae, loss_disc, gradients_disc, loss_real_disc, loss_fake_disc, loss_mixup_disc = self.train_step(batch)
                gradients_ae = np.mean([np.mean(i.numpy()) for i in gradients_ae])
                gradients_disc = np.mean([np.mean(i.numpy()) for i in gradients_disc])

                
                
                metrics_dict = {'Autoencoder Loss': loss_ae.numpy(), 
                        'Autoencoder Mean Gradient': gradients_ae,
                        'Autoencoder Reconstruction Loss': loss_reconstruction_ae.numpy(),
                        'Autoencoder Discrimination Loss': loss_discrimination_ae.numpy(),
                        'Autoencoder Mixup Loss': loss_mixup_ae.numpy(),
                        'Autoencoder Consistency Loss': loss_consistency_ae.numpy(),
                        'Discriminator Loss': loss_disc.numpy(), 
                        'Discriminator Mean Gradient': gradients_disc,
                        'Discriminator Real Loss': loss_real_disc.numpy(),
                        'Discriminator Fake Loss': loss_fake_disc.numpy(),
                        'Discriminator Mixup Loss': loss_mixup_disc.numpy()}
                    
                if log_wandb:
                    self.wandb_step(metrics_dict, epoch, plot = plot_data is not None, plot_data = plot_data)

                if verbose == 1:
                    self.progress_bar_step(progress_bar, step, metrics_dict, ['Autoencoder Loss','Discriminator Loss', 'Autoencoder Reconstruction Loss'])
                if verbose == 2:
                    self.progress_bar_step(progress_bar, step, metrics_dict)
                