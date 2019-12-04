import tensorflow as tf

class Autoencoder():
    def __init__(self, autoencoder_object, discriminator_object):
        self.AE = autoencoder_object
        self.Disc = discriminator_object

    def train_step_AE(self, batch):
        with tf.GradientTape() as tape:
            loss = self.AE.loss(batch, self.AE.autoencode(batch))
        gradients = tape.gradient(loss, self.AE.Model.trainable_variables)
        self.AE.Optimizer.apply_gradients(zip(gradients, self.AE.Model.trainable_variables))
        return loss

    def train_step_Disc(self, batch):
        with tf.GradientTape() as tape:
            disc_pred_real = self.Disc.discriminate(batch)
            disc_pred_fake = self.Disc.discriminate(self.AE.autoencode(batch))
            
            loss_real = self.Disc.loss(tf.ones(tf.shape(disc_pred_real)), disc_pred_real)
            loss_fake = self.Disc.loss(tf.zeros(tf.shape(disc_pred_fake)), disc_pred_fake)

            loss = loss_real + loss_fake
        gradients = tape.gradient(loss, self.Disc.Discriminator.trainable_variables)
        self.Disc.Optimizer.apply_gradients(zip(gradients, self.Disc.Discriminator.trainable_variables))
        return loss
    
    def train(self, epochs, dataset):
        for epoch in range(epochs):
            for batch in dataset:
                self.train_step_AE(batch)
                self.train_step_Disc(batch)
