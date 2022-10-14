from VIModels.utils import init_max
import tensorflow as tf 
import tensorflow.keras as tfk 
import tensorflow_probability as tfp 

tfkl = tfk.layers 
tfpl = tfp.layers 
tfd = tfp.distributions

class multihead_gumbel(tfk.Model):
    def __init__(self, config) -> None:
        super(multihead_gumbel, self).__init__()
        self.max_proba = init_max(hard=config.hard)

        self.encoder = config.encoder()
        self.decoders = [config.decoder(f'decoder_{i}') for i in range(config.num_heads)]
        self.heads = [config.head(f'head_{i}') for i in range(config.num_heads)]

    def call(self, input_tensor, training=False):
        x = input_tensor
        if training:
            encoding = self.encoder(x)
            samples = [decoder(encoding, training) for decoder in self.decoders]
            preds = [head(sample, training) for head, sample in zip(self.heads, samples)]
            return preds, encoding, self.max_proba(preds)
        preds = [head(x, training) for head in self.heads]
        return self.max_proba(preds)
        

class gumbel_classifier(tfk.Model):
    def __init__(self, config) -> None:
        super(gumbel_classifier, self).__init__()
        self.encoder = config.encoder()
        
        self.decoder = config.decoder()
        self.head = config.head()

    def call(self, input_tensor, training=False):
        x = input_tensor
        if training:
            encoding = self.encoder(x)
            samples = self.decoder(x, training)
            preds = self.head(samples, training)
            return preds, encoding
        preds = self.head(x, training)
        return preds