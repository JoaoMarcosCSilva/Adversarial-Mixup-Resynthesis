import tensorflow as tf
from tensorflow import keras
import numpy as np

from lib import baseline, mixup

class Autoencoder(baseline.Autoencoder):
    def __init__(self, Encoder, Decoder, Discriminator, Lambda, Gamma, Autoencoder_Optimizer = None, Discriminator_Optimizer = None, Mixup = mixup.interpolate):
        super().__init__(Encoder, Decoder, Discriminator, Autoencoder_Optimizer, Discriminator_Optimizer)
        self.Lambda = Lambda
        self.Gamma = Gamma
        self.Mixup = Mixup

    def interpolate_codes(self, codes):
        length = tf.shape(codes)[0]
        t = tf.random.uniform((length,1,1,1), 0, 0.5)
        codeI = self.Mixup(codes, codes[::-1], t)
        result = self.Decoder(codeI)
        return result, tf.reshape(t*2, (-1,1))
    
    def interpolate_batch(self, batch):
        length = tf.shape(batch)[0]
        t = tf.random.uniform((length,1,1,1), 0, 0.5)
        code1 = self.Encoder(batch)
        code2 = code1[::-1]
        codeI = self.Mixup(code1, code2, t)
        result = self.Decoder(codeI)
        return result, tf.reshape(t*2, (-1,1))
    
    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
            x_true = batch
            code = self.Encoder(batch)
            x_pred = self.Decoder(code) 

            x_mix,t = self.interpolate_codes(code)

            disc_pred_real = self.Discriminator(mixup.interpolate(x_true, x_pred, self.Gamma))
            disc_pred_fake = self.Discriminator(x_mix)

            loss_ae_reconstruction = self.autoencoder_loss(x_true, x_pred)
            loss_ae_discrimination = tf.reduce_mean(keras.losses.mse(tf.zeros_like(disc_pred_fake), disc_pred_fake))

            loss_disc_real = tf.reduce_mean(keras.losses.mse(tf.zeros_like(disc_pred_real), disc_pred_real))
            loss_disc_fake = tf.reduce_mean(keras.losses.mse(t, disc_pred_fake))           

            loss_ae = loss_ae_discrimination + self.Lambda*loss_ae_reconstruction
            loss_disc = loss_disc_fake + loss_disc_real
            
        ae_gradients = ae_tape.gradient(loss_ae, self.Autoencoder.trainable_variables)
        self.Autoencoder_Optimizer.apply_gradients(zip(ae_gradients, self.Autoencoder.trainable_variables))

        disc_gradients = disc_tape.gradient(loss_disc, self.Discriminator.trainable_variables)
        self.Discriminator_Optimizer.apply_gradients(zip(disc_gradients, self.Discriminator.trainable_variables))

        return loss_ae, ae_gradients, loss_ae_reconstruction, loss_ae_discrimination, loss_disc, disc_gradients, loss_disc_real, loss_disc_fake

    def train(self, epochs, dataset, verbose = 1, log_wandb = 0, plot_data = None, disc_steps = 1, ae_steps = 1):
        for epoch in range(epochs):
            if verbose:
                print('Epoch:',epoch+1)
                progress_bar = self.get_progress_bar(dataset, disc_steps + ae_steps)
            for step, batch in enumerate(dataset):
                if step % (disc_steps + ae_steps) == 0:
                    loss_disc, gradients_disc, loss_real_disc, loss_fake_disc = 0, 0, 0, 0
                    loss_ae, gradients_ae, loss_reconstruction_ae, loss_discrimination_ae = 0, 0, 0, 0
                loss_a, gradients_a, loss_reconstruction_a, loss_discrimination_a, loss_d, gradients_d, loss_real_d, loss_fake_d = self.train_step(batch)
                
                loss_ae += loss_a.numpy()
                gradients_ae += np.mean([np.mean(i.numpy()) for i in gradients_a])
                loss_reconstruction_ae += loss_reconstruction_a.numpy()
                loss_discrimination_ae += loss_discrimination_a.numpy()

                loss_disc += loss_d.numpy()
                gradients_disc += np.mean([np.mean(i.numpy()) for i in gradients_d])
                loss_real_disc += loss_real_d.numpy()
                loss_fake_disc += loss_fake_d.numpy()

                if (step + 1) % (disc_steps + ae_steps) == 0:
                    metrics_dict = {'Autoencoder Loss': loss_ae / ae_steps, 
                            'Autoencoder Mean Gradient': gradients_ae / ae_steps,
                            'Autoencoder Reconstruction Loss': loss_reconstruction_ae / ae_steps,
                            'Autoencoder Discrimination Loss': loss_discrimination_ae / ae_steps,
                            'Discriminator Loss': loss_disc / disc_steps, 
                            'Discriminator Mean Gradient': gradients_disc / disc_steps,
                            'Discriminator Real Loss': loss_real_disc / disc_steps,
                            'Discriminator Fake Loss': loss_fake_disc / disc_steps}
                    
                    if log_wandb:
                        self.wandb_step(metrics_dict, epoch, plot = plot_data is not None, plot_data = plot_data)

                    if verbose == 1:
                        self.progress_bar_step(progress_bar, step//(disc_steps + ae_steps), metrics_dict, ['Autoencoder Loss','Discriminator Loss', 'Autoencoder Reconstruction Loss'])
                    if verbose == 2:
                        self.progress_bar_step(progress_bar, step//(disc_steps + ae_steps), metrics_dict)
                