from typing import NamedTuple, Any
import tensorflow as tf
import tensorflow.keras as tfk 
import tensorflow.math as tfm
import tensorflow_probability as tfp 

tfd = tfp.distributions

class Config(NamedTuple):
    depth : Any 
    num_heads : Any 
    head : Any
    encoder : Any
    decoder : Any 

def init_sample_func(eps):
    def sample(logits, tau):
        shape = tfm.reduce_prod(logits.shape)
        gumbel = tfd.Gumbel(loc=0., scale=tf.ones(shape, dtype=tf.float32))
        y = logits + tf.reshape(gumbel.sample(), list(logits.shape))
        y = tfk.activations.softmax(y / tau)
        return y
    
    return sample

