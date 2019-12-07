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