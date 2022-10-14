from typing import NamedTuple, Any

import numpy as np

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

def init_max(hard=False):

    def hard_max_proba(self, proba):
        totals = tfm.reduce_sum(proba, axis=0)
        probs = tf.map_fn(lambda x : tf.one_hot(tf.argmax(x), depth=x.shape[-1], on_value=1, off_value=0, dtype=tf.float32), elems=totals)
        return probs

    def soft_max_proba(self, proba):
        totals = tfm.reduce_sum(proba, axis=0)
        norm=np.linalg.norm(totals)
        if norm==0:
            norm=np.finfo(totals.dtype).eps
        return totals / norm
    
    if hard:
        return hard_max_proba
    else:
        return soft_max_proba

def init_temp_anneal(init_tau, min_tau, rate):
    def temp_anneal(i):
        return np.maximum(init_tau*np.exp(-rate*i), min_tau)
    return temp_anneal

def multitask_loss(y_true, x_true, y_pred, p_x):
    cce = tfk.losses.CategoricalCrossentropy()
    intermediate = tfm.reduce_sum(tf.map_fn(lambda x : cce(y_true, x), elems=y_pred), axis=0, name="Sum of CE over Generated Preds")
    neg_log_likelihood = tf.reduce_sum(tf.map_fn(lambda x : tf.reduce_sum(x.log_prob(x_true),1), elems=p_x), axis=0, name="Sum of Neg Log Likelihood over each distribution")

    return intermediate + neg_log_likelihood