import tensorflow as tf
from ClassifierVAE.utils import compute_py 
from ClassifierVAE.structures import Decoder_Output, Encoder_Output
import tensorflow.keras as tfk 
import tensorflow_probability as tfp 

tfkl = tfk.layers 
tfpl = tfp.layers 
tfd = tfp.distributions


class encoder(tfkl.Layer):
    def __init__(self, config, **kwargs) -> None:
        super(encoder, self).__init__(name='encoder', **kwargs)
        self.n_class = config.n_class 
        self.n_dist = config.n_dist
        self.tau = config.tau
        self.encoder_stack = config.stack()
        self.dense_logits = tfkl.Dense(config.n_class * config.n_dist)
    
    def call(self, input_tensor, training=False):
        latent = self.encoder_stack(input_tensor)
        logits_y = self.dense_logits(latent)
        p_y = compute_py(logits_y, self.n_class, self.tau.read_value())   

        return Encoder_Output(tf.reshape(logits_y, [-1, self.n_dist, self.n_class]), p_y)

class decoder(tfkl.Layer):
    def __init__(self, config, **kwargs) -> None:
        super(decoder, self).__init__(**kwargs)
        self.gumbel = tfd.RelaxedOneHotCategorical # Gumbel-Softmax
        self.bernoulli = tfd.Bernoulli # Bernoulli for Reconstruction
        self.tau = config.tau # Temperature
        self.n_class = config.n_class # Number of Classes
        self.n_dist = config.n_dist # Number of Categorical Distributions

        self.decoder_stack = config.stack()
        self.reconstruct = tfkl.Dense(config.out_dim)

    def call(self, logits, training=False):
        q_y = self.gumbel(logits, self.tau.read_value())
        y = q_y.sample()

        decoded = self.decoder_stack(y)
        x_logits = self.reconstruct(decoded)

        p_x = self.bernoulli(logits=x_logits)
        x_mean = p_x.sample()

        return Decoder_Output(x_mean, y, p_x, q_y)

class head(tfkl.layer):
    def __init__(self, config, **kwargs) -> None:
        super(head, self).__init__(name='encoder', **kwargs)

        self.intermediate = config.intermediate() # Task Specific
        self.dense = config.stack()
        self.output = tfkl.Dense(config.n_class, activation='softmax')

    def call(self, input_tensor, training=False):
        latent = self.intermediate(input_tensor)
        dense = self.classification(latent)
        return self.output(dense)


def init_encoder(config, **kwargs):
    def new_encoder():
        return encoder(config, **kwargs)
    return new_encoder

def init_decoder(config, **kwargs):
    def new_decoder(name='decoder'):
        return encoder(config, name, **kwargs)
    return new_decoder

def init_head(config, **kwargs):
    def new_head(name='head'):
        return head(config, name, **kwargs)
    return new_head