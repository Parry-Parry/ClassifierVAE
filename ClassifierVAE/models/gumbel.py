from ClassiferVAE.utils import Encoder_Output, Model_Output, create_py, init_max
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

        self.n_class = config.n_class

    def call(self, input_tensor, training=False):
        x = input_tensor
        if training:
            encoder_output = self.encoder(x)
            decoder_outputs = [decoder(encoder_output.logits_y, training) for decoder in self.decoders]

            samples = [output.recons for output in decoder_outputs]
            preds = [head(sample, training) for head, sample in zip(self.heads, samples)]
            
            p_x = [output.p_x for output in decoder_outputs]
            q_y = [output.q_y for output in decoder_outputs]

            return Model_Output(preds, p_x, encoder_output.p_y, q_y, encoder_output.logits_y)
            
        preds = [head(x, training) for head in self.heads]

        return self.max_proba(preds)
        

class gumbel_classifier(tfk.Model):
    def __init__(self, config) -> None:
        super(gumbel_classifier, self).__init__()
        self.encoder = config.encoder()
        self.decoder = config.decoder()
        self.head = config.head()

        self.n_class = config.n_class

    def call(self, input_tensor, training=False):
        x = input_tensor
        if training:
            encoder_output = self.encoder(x)
            decoder_output = self.decoder(encoder_output.logits_y, training)
            preds = self.head(decoder_output.recons, training)
            return Model_Output(y_pred=preds, p_x=decoder_output.p_x, p_y=encoder_output.p_y, q_y=decoder_output.q_y, gen_y=encoder_output.logits_y)
        preds = self.head(x, training)

        return preds