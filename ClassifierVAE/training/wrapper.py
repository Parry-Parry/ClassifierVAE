import logging
from ClassifierVAE.utils import testing, init_max
import tensorflow as tf
import numpy as np

class wrapper():
    def __init__(self, proc) -> None:
        self.epochs = proc.epochs 

        self.model = proc.model 
        self.optim = proc.optim 
        self.loss = proc.loss

        if proc.temp:
            self.temp_anneal = proc.temp
        else:
            self.temp_anneal = None
        
        self.train_metric = proc.acc_metric() 
        self.val_metric = proc.acc_metric()
        self.maxi = init_max(False)
    
    def get_model(self):
        return self.model 

    def set_model(self, model):
        self.model = model

    def fit(self, train, test, logger):

        log = logging.getLogger(__name__)

        for epoch in range(self.epochs):

            log.info(f'Begin Epoch {epoch}...')

            for step, (x_batch, y_batch) in enumerate(train): 
                with tf.GradientTape() as tape:
                    output = self.model(x_batch, training=True)
                    loss_value = self.loss(y_batch, x_batch, output)
                print(len(self.model.trainable_weights))
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                print('grads computed')
                self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
                print('grads applied')
                self.train_metric.update_state(y_batch, self.maxi(output.y_pred))
        
            for x_batch, y_batch in test:
                test_pred = self.model(x_batch, training=False)
                self.val_metric.update_state(y_batch, test_pred)
            
            val_acc = self.val_metric.result()
            self.val_metric.reset_states()

            train_acc = self.train_metric.result()
            self.train_metric.reset_states()

            log.info(f'Epoch {epoch} | Train Acc: {train_acc} | Test Acc: {val_acc} | Loss: {loss_value}')

            logger.log({'epochs': epoch,
                   'loss': loss_value,
                   'acc': float(train_acc),
                   'val_acc': float(val_acc),
                   'temperature': self.temp_var})
            
            if self.temp_anneal:
                if step % self.step_anneal:
                    self.temp_var.assign(self.temp_anneal(step))
            
        results = testing(test, self.model, self.model.n_class)

        logger.log({'test_acc':results.acc, 'test_f1':np.mean(results.f1), 'test_rec':results.rec, 'test_prec':results.prec})

        return self.model
