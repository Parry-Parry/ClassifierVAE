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


def convnet(stack, kernel_size=5, strides=(2, 2), activation='relu'):
    return tfk.Sequential([tfkl.Conv2D(size, kernel_size=kernel_size, strides=strides, activation=activation) for size in stack])

def convtransposenet(stack, kernel_size=5, strides=(2, 2), activation='relu'):
    return tfk.Sequential([tfkl.Conv2DTranspose(size, kernel_size=kernel_size, strides=strides, activation=activation) for size in stack])

def densenet(stack, activation='relu'):
    return tfk.Sequential([tfkl.Conv2DTranspose(size, activation=activation) for size in stack])