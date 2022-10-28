from ClassifierVAE.utils import init_max
from ClassifierVAE.structures import Model_Output
import tensorflow as tf 
import tensorflow_probability as tfp 

tfk = tf.keras 
tfkl = tfk.layers 
tfpl = tfp.layers 
tfd = tfp.distributions

class multihead_gumbel(tfk.Model):
    def __init__(self, config) -> None:
        super(multihead_gumbel, self).__init__()
        self.input_layer = config.input_layer
        self.max_proba = init_max(hard=config.hard)

        self.encoder = config.encoder()
        self.decoders = [config.decoder(name=f'decoder_{i}') for i in range(config.num_heads)]
        self.heads = [config.head(name=f'head_{i}') for i in range(config.num_heads)]

        self.out_dim = config.out_dim
        self.n_class = config.n_class

    def call(self, input_tensor, training=False):
        x = input_tensor
        if self.input_layer: x = self.input_layer(x)
        if training:
            encoder_output = self.encoder(x, training)
            decoder_outputs = [decoder(encoder_output.logits_y, training) for decoder in self.decoders]
            samples = [output.recons for output in decoder_outputs]
            preds = [head(sample, training) for head, sample in zip(self.heads, samples)]
            x_logits = [output.x_logits for output in decoder_outputs]

            return Model_Output(preds, x_logits, encoder_output.gen_y)
            
        preds = [head(x, training) for head in self.heads]

        return self.max_proba(preds)
        

class gumbel_classifier(tfk.Model):
    def __init__(self, config) -> None:
        super(gumbel_classifier, self).__init__()
        self.input = config.inputs
        self.encoder = config.encoder()
        self.decoder = config.decoder()
        self.head = config.head()

        self.n_class = config.n_class

    def call(self, input_tensor, training=False):
        x = self.input(input_tensor)
        if training:
            encoder_output = self.encoder(x)
            decoder_output = self.decoder(encoder_output.logits_y, training)
            preds = self.head(decoder_output.recons, training)
            return Model_Output(y_pred=preds, p_x=decoder_output.p_x, p_y=encoder_output.p_y, q_y=decoder_output.q_y, gen_y=encoder_output.logits_y)
        preds = self.head(x, training)

        return preds