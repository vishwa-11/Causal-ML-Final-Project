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
    - NOTE: resorted to doing this due to keras insisting all labels must have a seperate cost fn
    """
    with np.load(file_path, allow_pickle=True) as data:
        dataset = tf.data.Dataset.from_tensor_slices(
            ({"text": tf.convert_to_tensor(data[text_key], name=text_key),
              "label": tf.convert_to_tensor(data[label_key], name=label_key, dtype=tf.int8), # cast to int
              "confounder": tf.convert_to_tensor(data[confounder_key], name=confounder_key, dtype=tf.int8)},
             tf.convert_to_tensor(data[label_key], name=label_key, dtype=tf.int8)))  # cast to int

    return dataset

class MMDRegularizerLayer(tf.keras.layers.Layer):
    """
    Layer that stores a MMD regularizing cost,
    Does nothing if mmd_loss_fn is None
    """
    def __init__(self, mmd_loss_fn: Callable = None,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.mmd_loss_fn = mmd_loss_fn
        self._call_fn = self.trivial_call if mmd_loss_fn is None else self.binded_call

    def call(self, y_pred, y, z):
        """
        Run the call function that was binded on init
        """
        return self._call_fn(y_pred, y, z)

    def binded_call(self, y_pred, y, z):
        """
        Call with mmd_loss_fn bindings
        """
        self.add_loss(self.mmd_loss_fn(y_pred, y, z))
        return y_pred

    def trivial_call(self, y_pred, y, z):
        """
        Call with no binds, trivially returning y_pred withou
        doing anythong
        """
        return y_pred


def build_augmented_model(preprocessing_layer: hub.KerasLayer,
                          bert_model: hub.KerasLayer,
                          mmd_loss_fn: Callable=None):
    """
    Create a base BERT model with its preprocessor as a model
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = preprocessing_layer(text_input)
    outputs = bert_model(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(2, activation=None, name='classifier')(net) # to get logP instead of logit

    label_input = tf.keras.layers.Input(shape=(), dtype=tf.int8, name="label")
    confounder_input = tf.keras.layers.Input(shape=(), dtype=tf.int8, name="confounder")
    net = MMDRegularizerLayer(mmd_loss_fn=mmd_loss_fn, name="mmd_loss")(net, label_input, confounder_input)

    return tf.keras.Model(inputs=[text_input, label_input, confounder_input], outputs=net)

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
                  M: float, N: float,
                  kernel: Callable= rbf_kernel) -> tf.Tensor:
    """
    Calculates the mmd using the given kernel function
    - Inputs:
        - x_i: (..., M, x_dim) tensor
        - x_j: (..., N, x_dim) tensor
        - M: size of x_i -> HACK to go around tf batch OBFUSICATION system
        - N: size of x_j -> HACK to go around tf batch OBFUSICATION system
        - kernel: Callabel that takes in (x_i, x_j) -> scalar
    - Outputs:
        - out: (...,) tensor
    """
    tf.Assert(
        tf.logical_and(tf.greater(tf.shape(x_i)[-2], 0), tf.greater(tf.shape(x_j)[-2],0)), [x_i, x_j]
    )
    tf.Assert(
        tf.logical_and(tf.greater(M, 0), tf.greater(N,0)), [x_i, x_j]
    )

    k_ii = tf.math.reduce_sum(kernel(x_i[...,:,None,:], x_i[...,None,:,:]), axis=(-1,-2)) / tf.cast(M*M, tf.float32)
    k_ij = tf.math.reduce_sum(kernel(x_i[...,:,None,:], x_j[...,None,:,:]), axis=(-1,-2)) / tf.cast(M*N, tf.float32)
    k_jj = tf.math.reduce_sum(kernel(x_j[...,:,None,:], x_j[...,None,:,:]), axis=(-1,-2)) / tf.cast(N*N, tf.float32)

    return tf.math.sqrt(k_ii - 2*k_ij + k_jj)

@tf.function
def marginal_mmd_loss(y_pred: tf.Tensor, y: tf.Tensor, z: tf.Tensor,
                         k: float=1.0,
                         kernel: Callable= rbf_kernel) -> tf.Tensor:
    """
    Calculates the conditional mmd[p(X|z=0), p(X|z=1)]
    - Inputs:
        - y: (B, y_dim) tensor, true labels
        - z: (B,) bool tensor, confounder value
        - y_pred: (B, y_dim) tensor, predicted labels
        - k: float, scaling factor
        - kernel: Callabel that takes in (x_i, x_j) -> scalar
    - Outputs:
        - out: () tensor
    """
    is_z0 = (z == 0)
    M = tf.reduce_sum(tf.cast(is_z0, tf.float32))
    N = tf.reduce_sum(tf.cast(~is_z0, tf.float32))
    return k * calculate_mmd(y_pred[is_z0], y_pred[~is_z0], M, N, kernel)

@tf.function
def conditional_mmd_loss(y_pred: tf.Tensor, y: tf.Tensor, z: tf.Tensor,
                         k: float=1.0,
                         kernel: Callable= rbf_kernel) -> tf.Tensor:
    """
    Calculates the conditional mmd[p(X|Z=0,Y=0), p(X|Z=1,Y=0)] + mmd[p(X|Z=0,Y=1), p(X|Z=1,Y=1)]
    - Inputs:
        - y: (B, y_dim) tensor, true labels
        - z: (B,) bool tensor, confounder value
        - y_pred: (B, y_dim) tensor, predicted labels
        - k: float, scaling factor
        - kernel: Callabel that takes in (x_i, x_j) -> scalar
    - Outputs:
        - out: () tensor
    """
    is_z0 = (z == 0)
    is_y0 = (y == 0)
    zy_00 = is_z0 & is_y0
    zy_10 = (~is_z0) & is_y0
    zy_01 = is_z0 & (~is_y0)
    zy_11 = (~is_z0) & (~is_y0)

    M1 = tf.reduce_sum(tf.cast(zy_00, tf.float32))
    N1 = tf.reduce_sum(tf.cast(zy_10, tf.float32))
    M2 = tf.reduce_sum(tf.cast(zy_01, tf.float32))
    N2 = tf.reduce_sum(tf.cast(zy_11, tf.float32))

    return k * calculate_mmd(y_pred[zy_00], y_pred[zy_10], M1, N1, kernel) + \
        k * calculate_mmd(y_pred[zy_01], y_pred[zy_11], M2, N2, kernel)