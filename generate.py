import numpy as np
import keras 
from keras.layers import *
from keras import Sequential
from keras.optimizers import Adam
from keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt

def build_generator():
    
    model = Sequential()
        
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=100))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(100,))
    img = model(noise)

    return Model(noise, img)


if __name__ == '__main__':
    generator = build_generator()
    generator.load_weights('%s-generator-%d.h5' % ('pokemon', 10000))
    noise = np.random.normal(-1, 1, 100)
    noise = np.array(noise, ndmin=2)
    random_number = np.random.randint(100)
    
    plt.imsave('new_pokemon-{}.png'.format(random_number), generator.predict(noise)[0])
    png = Image.open('new_pokemon-{}.png'.format(random_number), 'r')
    png.load() 
    png.resize((256,256), Image.ANTIALIAS).save('new_pokemon-{}.png'.format(random_number), quality=100)

