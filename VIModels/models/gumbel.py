import tensorflow as tf 
import tensorflow.keras as tfk 
import tensorflow_probability as tfp 

tfkl = tfk.layers 
tfpl = tfp.layers 
tfd = tfp.distributions

class multihead_gumbel(tfk.Model):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = config.encoder()
        self.heads = [config.head() for i in range(config.num_heads)]

class gumbel_classifier(tfk.Model):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = config.encoder()
        self.head = config.head()