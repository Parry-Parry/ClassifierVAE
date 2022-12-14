import tensorflow as tf
from ClassifierVAE.utils import compute_py, init_sample_func
from ClassifierVAE.structures import Decoder_Output, Encoder_Output
import tensorflow.keras as tfk 
import tensorflow_probability as tfp 

tfm = tf.math
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
        latent = self.encoder_stack(input_tensor, training)
        logits_y = tf.reshape(self.dense_logits(latent), [-1, self.n_dist, self.n_class])
        p_y = compute_py(logits_y, self.n_class, self.tau.read_value())   

        return Encoder_Output(logits_y, p_y)

class decoder(tfkl.Layer):
    def __init__(self, config, name='decoder', **kwargs) -> None:
        super(decoder, self).__init__(name=name, **kwargs)
        self.gumbel = init_sample_func(config.n_dist, config.n_class) # Gumbel-Softmax
        self.bernoulli = tfd.Bernoulli # Bernoulli for Reconstruction
        self.tau = config.tau # Temperature
        self.n_class = config.n_class # Number of Classes
        self.n_dist = config.n_dist # Number of Categorical Distributions
        self.out_dim = config.out_dim
        self.latent_square = config.latent_square

        self.process = tfkl.Dense(config.latent_square ** 2 * config.out_dim[-1], activation='relu')
        self.decoder_stack = config.stack()
        self.reconstruct = tfkl.Dense(tfm.reduce_prod(config.out_dim), activation='relu')

    def call(self, logits, training=False):
        y = self.gumbel(self.tau.read_value(), logits)
        processed_logits = tf.reshape(self.process(y), [-1, self.latent_square, self.latent_square, self.out_dim[-1]])
        decoded = self.decoder_stack(processed_logits, training)
        x_logits = self.reconstruct(decoded)

        p_x = self.bernoulli(logits=x_logits)
        x_mean = p_x.mean()

        return Decoder_Output(x_mean, x_logits, y)

class head(tfkl.Layer):
    def __init__(self, config, name='head', **kwargs) -> None:
        super(head, self).__init__(name=name, **kwargs)
        self.in_dim = config.in_dim
        self.intermediate = config.intermediate() # Task Specific
        self.stack = config.stack()
        self.clf = tfkl.Dense(config.n_class, activation='softmax')

    def call(self, input_tensor, training=False):
        x = tf.reshape(input_tensor, [-1,] + list(self.in_dim))
        latent = self.intermediate(x)
        dense = self.stack(latent)
        return self.clf(dense)


def init_encoder(config, **kwargs):
    def new_encoder():
        return encoder(config, **kwargs)
    return new_encoder

def init_decoder(config, **kwargs):
    def new_decoder(name='decoder'):
        return decoder(config, name, **kwargs)
    return new_decoder

def init_head(config, **kwargs):
    def new_head(name='head'):
        return head(config, name, **kwargs)
    return new_head