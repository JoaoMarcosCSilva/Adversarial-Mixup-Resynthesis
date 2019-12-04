from tensorflow import keras
import tensorflow as tf

def reconstruction_loss(y_true, y_pred):
    # y_true: the original data point x
    # y_pred: the data point x, reconstructed by the autoencoder

    return tf.reduce_mean(keras.losses.mse(y_true, y_pred))

def discriminator_loss(y_true, y_pred):
    # y_true: whether or not the point is fake
    # y_pred: the discrimator's prediction for that point
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))

def ae_gan_model_loss(y_true, y_pred, Discriminator, Lambda):
    # y_true: the original data point x
    # y_pred: the data point x, reconstructed by the autoencoder
    y_pred_disc = Discriminator(y_pred)
    y_true_disc = tf.ones(tf.shape(y_pred_disc))
    return Lambda*reconstruction_loss(y_true, y_pred) + discriminator_loss(y_true_disc, y_pred_disc)
    
def ae_gan_model_gradient(Encoder,Decoder, Discriminator, batch, Lambda):
    with tf.GradientTape() as tape:
        y_true = batch
        y_pred = Decoder(Encoder(batch))
        loss = ae_gan_model_loss(y_true, y_pred, Discriminator, Lambda)
    return loss, tape.gradient(loss, Encoder.trainable_variables + Decoder.trainable_variables)