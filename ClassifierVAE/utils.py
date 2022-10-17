from typing import NamedTuple, Any

import numpy as np

import tensorflow as tf
import tensorflow.keras as tfk 
import tensorflow.math as tfm
import tensorflow_probability as tfp 
import tensorflow_addons as tfa

tfd = tfp.distributions

class Model_Config(NamedTuple):
    num_heads : Any # How many decoder-classifer pairs
    encoder : Any # Encoder function
    decoder : Any # Decoder function
    head : Any # Classifier function
    n_class : Any # Number of Classes
    hard : Any # argmax (T) or softmax (F)

class Encoder_Config(NamedTuple):
    n_class : Any 
    n_dist : Any # Number of categorical distributions
    stack : Any # Dense sizes for encoder
    dense_activation : Any # Activation function
    tau : Any # Temperature tf variable

class Decoder_Config(NamedTuple):
    n_class : Any 
    n_dist : Any 
    stack : Any # Dense sizes for decoder
    dense_activation : Any
    tau : Any 

class Head_Config(NamedTuple):
    n_class : Any
    intermediate : Any # Task-specific layers
    stack : Any # Dense sizes for classifier
    dense_activation : Any

class Wrapper_Config(NamedTuple):
    model : Any 
    loss : Any 
    optim : Any 
    epochs : Any 
    temp : Any 
    acc_metric : Any 


class Encoder_Output(NamedTuple):
    logits_y : Any
    p_y : Any # Fixed Prior

class Decoder_Output(NamedTuple):
    recons : Any # Reconstruced x
    gen_y : Any # Generated Logits
    p_x : Any # Distribution over x
    q_y : Any # y Prior

class Model_Output(NamedTuple):
    y_pred : Any # Classifer Output
    p_x : Any # Reconstructed Distribition
    p_y : Any # Fixed Prior
    q_y : Any # Gumbel Prior
    gen_y : Any # Encoder Output

class Test_Results(NamedTuple):
    acc: Any
    f1: Any 
    rec: Any 
    prec: Any


def init_max(hard=False):
    
    def hard_max_proba(proba): # Takes argmax of summed preductions
        totals = tfm.reduce_sum(proba, axis=0)
        probs = tf.map_fn(lambda x : tf.one_hot(tf.argmax(x), depth=x.shape[-1], on_value=1, off_value=0, dtype=tf.float32), elems=totals)
        return probs
    
    def soft_max_proba(proba): # Normalizes summed probabilties to return a softmax
        totals = tfm.reduce_sum(proba, axis=0)
        norm=np.linalg.norm(totals)
        if norm==0:
            norm=np.finfo(totals.dtype).eps
        return totals / norm
    
    if hard:
        return hard_max_proba
    else:
        return soft_max_proba


def init_temp_anneal(init_tau, min_tau, rate): # Temperature annealing during training
    def temp_anneal(i):
        return np.maximum(init_tau*np.exp(-rate*i), min_tau)
    return temp_anneal


def compute_py(logits_y, n_class, tau): # Compute Fixed Prior
    logits_py = tf.ones_like(logits_y) * 1./n_class 
    return tfd.RelaxedOneHotCategorical(tau, logits=logits_py)


def init_loss(multihead=False):
    cce = tfk.losses.CategoricalCrossentropy()
    def ensemble_loss(y_true, x_true, output):
        qp_pairs = [q_y.log_prob(output.gen_y) - output.p_y.log_prob(output.gen_y) for q_y in output.q_y]
        KL = tf.reduce_sum([tf.reduce_sum(qp, 1) for qp in qp_pairs], axis=0, name="Sum of KL over Prior Distribution and Learned Distributions")

        intermediate = tfm.reduce_sum(tf.map_fn(lambda x : cce(y_true, x), elems=output.y_pred), axis=0, name="Sum of CE over Generated Preds")
        neg_log_likelihood = tf.reduce_sum(tf.map_fn(lambda x : tf.reduce_sum(x.log_prob(x_true), 1), elems=output.p_x), axis=0, name="Sum of Neg Log Likelihood over each distribution")

        return intermediate + neg_log_likelihood - KL
    
    def sequential_loss(y_true, x_true, output):
        qp = output.q_y.log_prob(output.gen_y) - output.p_y.log_prob(output.gen_y) 
        KL = tf.reduce_sum(qp, 1)

        intermediate = cce(y_true, output.y_pred)
        neg_log_likelihood = tf.reduce_sum(output.p_x.log_prob(x_true), 1)

        return intermediate + neg_log_likelihood - KL
    
    if multihead: return ensemble_loss
    return sequential_loss

def testing(test_set, model, n_classes=10):
    test_acc_metric = tfk.metrics.CategoricalAccuracy()
    test_f1_metric = tfa.metrics.F1Score(num_classes=n_classes)
    test_recall_metric = tfk.metrics.Recall()
    test_precision_metric = tfk.metrics.Precision()

    for x_batch, y_batch in test_set:
        test_pred = model(x_batch, training=False)
        test_acc_metric.update_state(y_batch, test_pred)
        test_f1_metric.update_state(y_batch, test_pred)
        test_recall_metric.update_state(y_batch, test_pred)
        test_precision_metric.update_state(y_batch, test_pred)

    results = Test_Results(test_acc_metric.result(), test_f1_metric.result(), test_recall_metric.result(), test_precision_metric.result())

    return results