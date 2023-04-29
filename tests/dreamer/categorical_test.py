import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.layers import Lambda


def test_argmax():
    batch_size = 64
    num_classes = 10
    ohehot_categorical_input = tf.zeros((batch_size, 32, num_classes))

    argmax = Lambda(lambda x: tf.cast(tf.argmax(
        tf.cast(x, dtype=tf.float32), axis=-1), dtype=tf.float32))
    output = argmax(ohehot_categorical_input)

    assert output.shape == (batch_size, 32)


def test_onehot():
    batch_size = 64
    categorical_input = tf.zeros((batch_size, 32, 16))

    sample = tfp.layers.OneHotCategorical((32, 16))
    output = sample(categorical_input)

    assert output.shape == (batch_size, 32, 16)
    assert np.all(tf.reduce_sum(output, axis=-1) == tf.ones((batch_size, 32)))
