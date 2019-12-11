import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lib.mixup import interpolate



def get_wandb_images(Autoencoder, rows, columns, data, seed = 1):
  if seed != None:
    np.random.seed(seed)

  images = data[np.random.choice(range(len(data)), size = rows*2, replace = False)]  
  codes = Autoencoder.Encoder(images)
  
  code_shape = tuple(tf.shape(codes)[1:].numpy())
  to_be_decoded = np.zeros((rows*columns,) + code_shape)
  
  for row in range(rows):
    for column in range(columns):
        to_be_decoded[row*columns + column] = interpolate(codes[row], codes[-row+1], column/(columns-1))
      
  decoded_images = Autoencoder.Decoder(to_be_decoded).numpy()
  return decoded_images
  

def get_output_image(Autoencoder, rows, columns, data, seed = -1, plot = True):
  if plot:
    plt.ion()
  else:
    plt.ioff()
    
  if seed != -1:
    np.random.seed(seed)
  
  plt.figure(figsize = (18/12*columns,18/12*(rows+0.1)))

  for count in range(rows):
    i = np.random.randint(0,len(data))
    j = np.random.randint(0,len(data))

    code1 = Autoencoder.Encoder.predict((data[i:i+1]))
    code2 = Autoencoder.Encoder.predict((data[j:j+1]))

    plt.subplot(rows,columns,count*columns + 1)
    if Autoencoder.Discriminator != None:
      plt.title(Autoencoder.Discriminator.predict(data[i:i+1]))
    plt.imshow(data[i])
  
    for im in range (columns-2):
      t = im/(columns-3)
      plt.subplot(rows,columns,count*columns+im+2)

      code = interpolate(code1, code2, t)
      image = Autoencoder.Decoder.predict(code)[0]
      
      if Autoencoder.Discriminator != None:
        plt.title(Autoencoder.Discriminator.predict(image.reshape(1,64,64,3))[0])
      plt.imshow(image)

    plt.subplot(rows,columns,(count+1)*columns)
    if Autoencoder.Discriminator != None:
      plt.title(Autoencoder.Discriminator.predict(data[j:j+1]))
    plt.imshow(data[j])


