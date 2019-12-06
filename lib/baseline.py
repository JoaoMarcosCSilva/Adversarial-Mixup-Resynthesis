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
        return loss.numpy(), np.array(gradients)

    def train(self, epochs, dataset, verbose = True, wandb_run = False):
        for epoch in range(epochs):
            if verbose:
                print('Epoch:',epoch+1)
            for batch in dataset:
                loss, gradients = self.train_step_AE(batch)
            if wandb_run:
                wandb.log({'Epoch': epoch}, commit = False)
                wandb.log({'Autoencoder Loss': loss, 
                    'Autoencoder Mean Gradient': np.mean([np.mean(i) for i in gradients]))
