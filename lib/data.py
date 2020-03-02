import os
import imageio
import glob

import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import tqdm
import requests
import zipfile
from tqdm.autonotebook import tqdm

def get_kaggle_json_dict(filepath):
    file = open(filepath)
    kaggle_json = json.load(file)
    file.close()
    return kaggle_json
    

def download_shoes_data():
    url = 'http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip'
    myfile = requests.get(url)
    open('Data/shoes.zip', 'wb').write(myfile.content)
    with zipfile.ZipFile('Data/shoes.zip', 'r') as zip_ref:
        zip_ref.extractall('Data/shoes/')

def load_shoes_data(verbose = 1):
    pathfiles = glob.glob('Data/shoes/**/*.jpg', recursive = True)
    images = []
    
    if verbose:
        pathfiles = tqdm(pathfiles)
    for path in pathfiles:
        im = imageio.imread(path)
        if im.shape == (136,136,3):
            images.append(im)

    images = np.array(images, dtype = np.float32)

    images = images/255

    images = np.float32(images)

    x_train, x_test = train_test_split(images, test_size = 0.1, random_state = 1)

    return x_train, x_test

def download_pokemon_data (filepath):
    kaggle_json_dict = get_kaggle_json_dict(filepath)
    os.environ['KAGGLE_USERNAME'] = kaggle_json_dict['username'] # username from the json file 
    os.environ['KAGGLE_KEY'] = kaggle_json_dict['key'] # key from the json file

    try:
        import kaggle
    except:
        print('Please install the kaggle library')


    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('brilja/pokemon-mugshots-from-super-mystery-dungeon',path = 'Data', unzip=True)

def load_pokemon_data ():
    images = []

    for im_path in glob.glob('Data/smd/**/*.png', recursive = True):
        images.append(imageio.imread(im_path))
    images = np.array(images)

    images = images / 255
    
    images = np.float32(images)

    x_train, x_test = train_test_split(images, test_size = 0.1, random_state = 1)

    return x_train, x_test

def augment_data (data, multiplier):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
    )

    augmented_data = data

    for i in datagen.flow(data, batch_size=len(data)):
        augmented_data = np.concatenate((augmented_data, i))
        break
    
    return augmented_data

def get_tf_data(data, batch_size = 32, shuffle_size = 100):
    return tf.data.Dataset.from_tensor_slices(data).shuffle(shuffle_size).batch(batch_size)