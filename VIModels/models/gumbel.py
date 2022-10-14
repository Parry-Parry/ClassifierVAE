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
        self.decoders = [config.decoder(f'decoder_{i}') for i in range(config.num_heads)]
        self.heads = [config.head(f'head_{i}') for i in range(config.num_heads)]
        

class gumbel_classifier(tfk.Model):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = config.encoder()
        self.decoder = config.decoder()
        self.head = config.head()