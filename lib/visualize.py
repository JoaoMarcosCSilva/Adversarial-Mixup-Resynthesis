import numpy as np
from lib.mixup import interpolate
import matplotlib.pyplot as plt

def get_output_image(Autoencoder, rows, columns, data, filename = None, seed = -1, interpolate_code = True, pass_again = False, disc = True):
  if seed != -1:
    np.random.seed(seed)
  plt.figure(figsize = (18/12*columns,18/12*rows))

  if not interpolate_code:
    for count in range(rows):
      i = np.random.randint(0,len(data))
      j = np.random.randint(0,len(data))

      image1 = Autoencoder.AE.Model.predict(data[i:i+1])[0]
      image2 = Autoencoder.AE.Model.predict(data[j:j+1])[0]

      plt.subplot(rows,columns,count*columns + 1)
      plt.imshow(data[i])
      
      for im in range (columns-2):
        t = im/(columns-3)

        plt.subplot(rows,columns,count*columns+im+2)
    
        image = interpolate(image1, image2, t)
        
        plt.imshow(image)

      plt.subplot(rows,columns,(count+1)*columns)
      plt.imshow(data[j])

  else:
    for count in range(rows):
      i = np.random.randint(0,len(data))
      j = np.random.randint(0,len(data))

      code1 = Autoencoder.AE.Encoder.predict((data[i:i+1]))
      code2 = Autoencoder.AE.Encoder.predict((data[j:j+1]))

      plt.subplot(rows,columns,count*columns + 1)
      if disc:
        plt.title(Autoencoder.Disc.Discriminator.predict(data[i:i+1]))
      plt.imshow(data[i])
     
      for im in range (columns-2):
        t = im/(columns-3)
        plt.subplot(rows,columns,count*columns+im+2)

        code = interpolate(code1, code2, t)
        image = Autoencoder.AE.Decoder.predict(code)
        if pass_again:
          image = Autoencoder.AE.Model.predict(image)[0]
        else:
          image = image[0]
        if disc:
          plt.title(Autoencoder.Disc.Discriminator.predict(image.reshape(1,64,64,3))[0])
        plt.imshow(image)

      plt.subplot(rows,columns,(count+1)*columns)
      if disc:
        plt.title(Autoencoder.Disc.Discriminator.predict(data[j:j+1]))
      plt.imshow(data[j])

  if filename is not None:
    plt.savefig(filename)
  
  plt.show()