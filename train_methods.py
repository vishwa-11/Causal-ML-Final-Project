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
              "label": tf.convert_to_tensor(data[label_key], name=label_key),
              "confounder": tf.convert_to_tensor(data[confounder_key], name=confounder_key)},
             tf.convert_to_tensor(data[label_key], name=label_key)))

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
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

    label_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="label")
    confounder_input = tf.keras.layers.Input(shape=(), dtype=tf.bool, name="confounder")
    net = MMDRegularizerLayer(mmd_loss_fn=mmd_loss_fn, name="mmd_loss")(net, label_input, confounder_input)

    return tf.keras.Model(inputs=[text_input, label_input, confounder_input], outputs=net)