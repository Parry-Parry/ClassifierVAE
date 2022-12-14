import collections
from pathlib import Path, PurePath
import pickle

import tensorflow as tf
import tensorflow.keras as tfk 
import tensorflow.math as tfm
import tensorflow_probability as tfp 
import tensorflow_addons as tfa

from tensorflow.keras.datasets import cifar100, cifar10, mnist
from ClassifierVAE.structures import Test_Results, Dataset
import numpy as np
from sklearn.cluster import KMeans

tfd = tfp.distributions
tfb = tfp.bijectors

'''
Gumbel Softmax (No tfd)
'''
def init_sample_func(n_dist, n_class):
    def sample(tau, logits_y):
        U = tf.random.uniform(logits_y.shape, 0, maxval=1)
        y = logits_y - tfm.log(-tfm.log(U + 1e-20) + 1e-20) # logits + gumbel noise
        y = tfk.activations.softmax(tf.reshape(y, [-1, n_dist, n_class]) / tau)
        y = tf.reshape(y, [-1, n_dist*n_class])
        return y
    return sample

'''
Function to prevent NaN values
'''

def nan_remove(x):
    return tf.where(tfm.is_nan(x), tf.constant(-np.inf, dtype=x.dtype), x)

def prob_diff(d1, d2, x):
    diff = nan_remove(d1.log_prob(x)) - nan_remove(d2.log_prob(x))
    return nan_remove(diff)

'''
Creates a function which recieves a [num_distribution x n_class] tensor of probabilities, then takes either the argmax or softmax (normalized) of that sum
'''

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

'''
Initializes the temperature anneal callback with hyperparameters
'''

def init_temp_anneal(init_tau, min_tau, rate): # Temperature annealing during training
    def temp_anneal(i):
        return np.maximum(init_tau*np.exp(-rate*i), min_tau)
    return temp_anneal

'''
Computes the latent fixed prior over the logits of y
'''

def compute_py(logits_y, n_class, tau): 
    logits_py = tf.ones_like(logits_y) * 1./n_class 
    return tfd.RelaxedOneHotCategorical(tau, logits=logits_py)

'''
Initializes multitask loss with the sum taken over ensemble components


def init_loss(multihead=False):
    cce = tfk.losses.CategoricalCrossentropy()
    def ensemble_loss(y_true, x_true, output):
        qp_pairs = [prob_diff(q_y, output.p_y, output.gen_y) for q_y in output.q_y]
        KL = [nan_remove(tfm.reduce_mean(qp)) for qp in qp_pairs]
        neg_ll = [tf.reduce_mean(tf.reduce_sum(-x.log_prob(tf.reshape(x_true, [x_true.shape[0], -1])), 1)) for x in output.p_x]
        elbo = tf.reduce_sum([tf.reduce_mean(ll - kl) for ll, kl in zip(neg_ll, KL)])

        intermediate = tfm.reduce_sum([cce(y_true, x) for x in output.y_pred], axis=0, name='Sum of CCE over each head predictions')
        
        return intermediate - elbo
    
    def sequential_loss(y_true, x_true, output):
        KL = tfm.reduce_mean(prob_diff(output.q_y, output.p_y, output.gen_y))
        neg_log_likelihood = tfm.reduce_sum(output.p_x.log_prob(tf.reshape(x_true, [x_true.shape[0], -1])), 1)
        elbo = tfm.reduce_mean(neg_log_likelihood - KL)

        intermediate = cce(y_true, output.y_pred) 

        return intermediate - elbo
    
    if multihead: return ensemble_loss
    return sequential_loss
'''
def init_loss(multihead, n_dist, n_class):
    cce = tfk.losses.CategoricalCrossentropy()
    bce = tfk.losses.BinaryCrossentropy()
    softmax = tfk.activations.softmax

    def ensemble_loss(y_true, x_true, output):
        q_y = softmax(tf.reshape(output.gen_y, [-1, n_dist, n_class]))
        log_q_y = tfm.log(q_y + 1e-20)
        kl_tmp = q_y * (log_q_y - tfm.log(1.0/n_class))
        KL = tfm.reduce_mean(tf.reduce_sum(kl_tmp, axis=(1, 2)))
        
        elbo = tf.reduce_sum([tfm.reduce_prod(x_true[1:]) * bce(tf.reshape(x_true, [x_true.shape[0], -1]), logits) for logits in output.x_logits]) - KL 
        intermediate = tf.reduce_sum([cce(y_true, x) for x in output.y_pred], axis=0, name='Sum of CCE over each head predictions')
        
        return intermediate + elbo
    
    def sequential_loss(y_true, x_true, output):
        KL = tfm.reduce_mean(prob_diff(output.q_y, output.p_y, output.gen_y))
        neg_log_likelihood = tfm.reduce_sum(output.p_x.log_prob(tf.reshape(x_true, [x_true.shape[0], -1])), 1)
        elbo = tfm.reduce_mean(neg_log_likelihood - KL)

        intermediate = cce(y_true, output.y_pred) 
        return intermediate - elbo
    
    if multihead: return ensemble_loss
    return sequential_loss


'''
Runs tests on standard metrics
'''

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

'''
Retrieve and normalize image datasets
'''

def retrieve_dataset(name=None, path=None):
    normalize = lambda w, x, y, z : (w / np.float32(255), x / np.float32(255), y.astype(np.int64), z.astype(np.int64))
    if path: 
        """Not Implemented until testing complete on standard datasets"""
        (x_train, y_train), (x_test, y_test) = None
        return PurePath(path).parent.name, normalize(x_train, x_test, y_train, y_test)
    if name:
        if name == 'MNIST':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif name == 'CIFAR10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif name == 'CIFAR100':
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        else:
            return None
        return name, normalize(x_train, x_test, y_train, y_test)
    return None, None

'''
Retrieves stored aggregate datasets by number of clusters K or generates a new aggregate set
'''

def aggregate(data, K, dir, seed):
    pure = PurePath(dir)
    path = pure.joinpath(data.name + str(K) + str(seed) + '.pkl')

    if Path(path).exists():
        with open(path, 'rb') as f:
            tmp = pickle.load(f)
            aggr_x, aggr_y, avg = tmp
            shape = aggr_x.shape
    else:
        shape = tuple([K] + list(data.x_train.shape[1:]))
        x = np.array([img.flatten() for img in data.x_train])
        
        if not seed: seed = np.random.randint(9999)
        clustering = KMeans(n_clusters=K, random_state=seed).fit_predict(x)

        cluster_members =  collections.defaultdict(list)
        cluster_labels = collections.defaultdict(list)
        for a, b, c in zip(x, data.y_train, clustering): 
            cluster_members[c].append(a)
            if 'CIFAR' in data.name:
                cluster_labels[c].append(b[0])
            else:
                cluster_labels[c].append(b)
        
        centroids = []
        labels = []
        member_count = []

        for k, v in cluster_members.items():
            centroids.append(np.mean(v, axis=0))
            vals, counts = np.unique(cluster_labels[k], return_counts=True)
            labels.append(vals[np.argmax(counts)]) # majority class
            member_count.append(len(v))
        
        aggr_x = np.reshape(np.array(centroids), shape)
        aggr_y = np.array(labels)
        avg = np.mean(member_count)

        with open(path, 'wb') as f:
            pickle.dump((aggr_x, aggr_y, avg), f)

    return Dataset(data.name, aggr_x, data.x_test, aggr_y, data.y_test)

def build_dataset(dataset, K, path, batch_size, seed):
    if K != 1:
        dataset = aggregate(dataset, K, path, seed)

    if dataset.name == 'CIFAR100':
        n_classes = 100
    else:
        n_classes = len(np.unique(dataset.y_train))

    if len(dataset.x_train.shape) == 3:
        x_train = np.expand_dims(dataset.x_train, axis=-1)
        x_test = np.expand_dims(dataset.x_test, axis=-1)
    else:
        x_train = dataset.x_train
        x_test = dataset.x_test

    y_train = tfk.utils.to_categorical(dataset.y_train, n_classes)
    y_test = tfk.utils.to_categorical(dataset.y_test, n_classes)

    tf_convert = lambda x, y, types : (tf.data.Dataset.from_tensor_slices((tf.cast(x, types[0]), tf.cast(y, types[1])))).shuffle(len(dataset.x_train)).batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)

    train_set = tf_convert(x_train, y_train, [tf.float32, tf.uint8])
    test_set = tf_convert(x_test, y_test, [tf.float32, tf.uint8])

    return train_set, test_set, n_classes
