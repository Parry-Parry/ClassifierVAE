from typing import NamedTuple, Any
import tensorflow as tf
import tensorflow.keras as tfk 
import tensorflow.math as tfm
import tensorflow_probability as tfp 

tfd = tfp.distributions

class Model_Config(NamedTuple):
    num_heads : Any 
    encoder : Any
    decoder : Any
    head : Any

class Encoder_Config(NamedTuple):
    n_class : Any
    n_dist : Any 
    stack : Any
    dense_activation : Any

class Decoder_Config(NamedTuple):
    n_class : Any
    n_dist : Any 
    stack : Any
    tau : Any
    dense_activation : Any

class Head_Config(NamedTuple):
    dense_activation : Any
