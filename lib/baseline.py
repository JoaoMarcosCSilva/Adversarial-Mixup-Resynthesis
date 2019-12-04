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

    def train(self, epochs, dataset):
        for epoch in range(epochs):
            for batch in dataset:
                self.train_step_AE(batch)
