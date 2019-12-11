import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from lib.mixup import interpolate

def get_wandb_plot(Autoencoder, rows, columns, dataset, seed = -1):
    if seed != -1:
        np.random.seed(seed)
    fig = Figure(figsize = (18/12*columns,18/12*(rows+0.1)))
    ax = fig.subplots(rows, columns)
    
    add_title = Autoencoder.Discriminator is not None
    
    images = dataset[np.random.choice(range(len(dataset)), size = rows*2, replace = False)]  
    codes = Autoencoder.Encoder.predict(images)
      
    images_1 = images[:rows]
    images_2 = images[rows:]
    codes_1 = codes[:rows]
    codes_2 = codes[rows:]
    
    for row in range(rows):
        for column in range(columns):
            if column == 0:
                if add_title:
                    ax[row][column].set_title(Autoencoder.Discriminator.predict(images_1[row:row+1]))
                ax[row][column].imshow(images_1[row])
            elif column == columns - 1:
                if add_title:
                    ax[row][column].set_title(Autoencoder.Discriminator.predict(images_2[row:row+1]))
                ax[row][column].imshow(images_2[row])
            else:
                alpha = column/(columns+1)                
                code = interpolate(codes_1, codes_2, alpha)
                image = Autoencoder.Decoder.predict(tf.reshape(code, (1,) + tf.shape(code)))
                if add_title:
                    ax[row][column].set_title(Autoencoder.Discriminator.predict(image))
                ax[row][column].imshow(image[0])
      
    return fig
  

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


