import tensorflow as tf 
import tensorflow.keras as tfk 
import tensorflow_probability as tfp 

tfkl = tfk.layers 
tfpl = tfp.layers 
tfd = tfp.distributions

'''
def init_generator(sampling_func, depth, out_dim):
    def generator():
        return tfk.Sequential([
            tfkl.Lambda(sampling_func, output_shape=(M*N,))(logits_y)
        ])
    return generator
'''
def init_encoder():
    def encoder():
        pass 
    return encoder
def init_head():
    def head():
        pass 
    return head