import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Callable, Iterable


def load_dataset(file_path: str,
                 text_key: str, confounder_key: str,
                 label_key: str):
    """
    Returns a tensorflow dataset with
    - X: text inputs
    - Z: confounder inputs
    - Y: label ouputs
    """
    with np.load(file_path, allow_pickle=True) as data:
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(data[text_key], name=text_key),
             {"label": tf.convert_to_tensor(data[label_key], name=label_key),
              "confounder": tf.convert_to_tensor(data[confounder_key], name=confounder_key)}))

    return dataset

class MMDLoss(tf.keras.losses.Loss):
    """
    Wrapper class for MMD regularizing losses
    """
    def __init__(self, mmd_loss_fn: Callable,
                 label_key='label',confounder_key='confounder'):
        super().__init__()
        self.mmd_loss_fn = mmd_loss_fn
        self.label_key = label_key
        self.confounder_key = confounder_key

    def call(self, y_true, y_pred):
        return self.mmd_loss_fn(y_true[self.label_key], y_true[self.confounder_key],
                                y_pred)

class AugmentedModel(tf.keras.Model):
    """
    An augmented to handle customized calls between multiple labels to single output scenario
    """
    def __init__(self, bert_model,
                 label_loss: tf.keras.losses.Loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 label_metric: tf.keras.metrics.Metric=tf.keras.metrics.BinaryAccuracy(),
                 mmd_losses: Iterable[MMDLoss]= [],
                 *args, **kwargs):
        super(AugmentedModel, self).__init__(*args, **kwargs)
        self.model = bert_model
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.label_loss = label_loss
        self.label_metric = label_metric
        self.mmd_losses = mmd_losses

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    def compute_loss(self, x, y, y_pred, sample_weight=None):
        for mmd_loss in self.mmd_losses:
            print(mmd_loss(y_true=y, y_pred=y_pred))
        # sample_weight = tf.constant([1.0], dtype=tf.float32) if sample_weight is None else sample_weight
        loss_val = self.label_loss(y_true=y['label'], y_pred=y_pred)
        for mmd_loss in self.mmd_losses:
            loss_val += mmd_loss(y_true=y, y_pred=y_pred)
        self.loss_tracker.update_state(loss_val)
        return loss_val

    def compute_metrics(self, x, y, y_pred, sample_weight):
        self.label_metric.update_state(y_true=y['label'], y_pred=y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.label_metric.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker, self.label_metric]

def build_augmented_model(preprocessing_layer: hub.KerasLayer,
                          bert_model: hub.KerasLayer,
                          mmd_loss_fns: Iterable[Callable]=[])->tf.keras.Model:
    """
    Create a base BERT model with its preprocessor as a model
    """
    # create the functional layer
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = preprocessing_layer(text_input)
    outputs = bert_model(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

    model = tf.keras.Model(inputs=text_input, outputs=net)

    # return augmented model
    return AugmentedModel(bert_model=model,
                          mmd_losses=[MMDLoss(mmd_loss_fn=mmd_loss_fn) for mmd_loss_fn in mmd_loss_fns])

@tf.function
def rbf_kernel(x_i: tf.Tensor, x_j: tf.Tensor,
               bandwidth: float = 10.) -> tf.Tensor:
    """
    Rbf kernel given to similar sized tensor
    - Inputs:
        - x_i: (..., x_dim) tensor
        - x_j: (..., x_dim) tensor
        - bandwidth: float
    - Outputs:
        - out: (...,) tensor
    """
    delta = x_i - x_j
    dist = (delta[...,None,:] @ delta[...,:,None])[...,0,0]
    return tf.math.exp(-dist/(2*bandwidth**2))

@tf.function
def calculate_mmd(x_i: tf.Tensor, x_j: tf.Tensor,
                  kernel: Callable= rbf_kernel) -> tf.Tensor:
    """
    Calculates the mmd using the given kernel function
    - Inputs:
        - x_i: (..., M, x_dim) tensor
        - x_j: (..., N, x_dim) tensor
        - kernel: Callabel that takes in (x_i, x_j) -> scalar
    - Outputs:
        - out: (...,) tensor
    """
    M = tf.shape(x_i)[-2]
    N = tf.shape(x_j)[-2]

    tf.Assert(
        tf.logical_and(tf.greater(M, 0), tf.greater(N,0)), [x_i, x_j]
    )

    k_ii = tf.math.reduce_sum(kernel(x_i[...,:,None,:], x_i[...,None,:,:]), axis=(-1,-2)) / tf.cast(M*M, tf.float32)
    k_ij = tf.math.reduce_sum(kernel(x_i[...,:,None,:], x_j[...,None,:,:]), axis=(-1,-2)) / tf.cast(M*N, tf.float32)
    k_jj = tf.math.reduce_sum(kernel(x_j[...,:,None,:], x_j[...,None,:,:]), axis=(-1,-2)) / tf.cast(N*N, tf.float32)
    return tf.math.sqrt(k_ii - 2*k_ij + k_jj)

@tf.function
def conditional_mmd_loss(y: tf.Tensor, z: tf.Tensor, y_pred: tf.Tensor,
                         k: float=1.0,
                         kernel: Callable= rbf_kernel) -> tf.Tensor:
    """
    Calculates the conditional mmd[p(X|z=0), p(X|z=1)]
    - Inputs:
        - y: (B, y_dim) tensor, true labels
        - z: (B, 2) bool tensor, confounder value
        - y_pred: (B, y_dim) tensor, predicted labels
        - k: float, scaling factor
        - kernel: Callabel that takes in (x_i, x_j) -> scalar
    - Outputs:
        - out: () tensor
    """
    is_z0 = (z == 0)
    return k * calculate_mmd(y_pred[is_z0], y_pred[~is_z0], kernel)