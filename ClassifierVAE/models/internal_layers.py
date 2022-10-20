import tensorflow as tf 

tfk = tf.keras
tfkl = tfk.layers

'''
Set of convenience functions to initialize internal structures depending on function

args:
    stack : list(int) -> Set of output sizes
    kernel_size : int -> Kernel size for convolutions
    strides : tuple(int) -> Size of convolutional strides
    activation : Activation function for each layer
returns:
    tfk.Sequential -> Simple sequential keras structure
'''

def init_convnet(stack, kernel_size=5, strides=(2, 2), activation='relu', dropout_rate=None, flatten=False):
    def convnet():
        if dropout_rate: 
            layers = []
            for size in stack:
                layers.append(tfkl.Conv2D(size, kernel_size=kernel_size, strides=strides, activation=activation))
                layers.append(tfkl.Dropout(dropout_rate))
            layers = layers[:-1]
        else:
            layers = [tfkl.Conv2D(size, kernel_size=kernel_size, strides=strides, activation=activation) for size in stack]
        if flatten: layers = layers + [tfkl.flatten()]
        return tfk.Sequential(layers)
    return convnet 

def init_convtransposenet(stack, kernel_size=5, strides=(2, 2), activation='relu', dropout_rate=None, flatten=False):
    def convtransposenet():
        if dropout_rate: 
            layers = []
            for size in stack:
                layers.append(tfkl.Conv2DTranspose(size, kernel_size=kernel_size, strides=strides, activation=activation))
                layers.append(tfkl.Dropout(dropout_rate))
            layers = layers[:-1]
        else:
            layers = [tfkl.Conv2DTranspose(size, kernel_size=kernel_size, strides=strides, activation=activation) for size in stack]
        if flatten: layers = layers + [tfkl.flatten()]
        return tfk.Sequential(layers)
    return convtransposenet

def init_densenet(stack, activation='relu', dropout_rate=None):
    def densenet():
        if dropout_rate:
            layers = []
            for size in stack:
                layers.append(tfkl.Dense(size, activation=activation))
                layers.append(tfkl.Dropout(dropout_rate))
            layers = layers[:-1]
        else:
            layers = [tfkl.Dense(size, activation=activation) for size in stack]
        return tfk.Sequential(layers)
    return densenet
