import numpy as np
import tensorflow as tf
from algos.dreamer.layers import VQCategorical, VQCodebook


def test_all_shapes_fine():
    batch_size = 128
    num_classifications = 16
    num_classes = 32
    inputs = tf.random.normal((batch_size, 8, 8, 64))

    vq_codebook = VQCodebook(num_classifications, num_classes)
    vq_cat = VQCategorical(vq_codebook)
    vq_cat.build(inputs.shape)

    categoricals = vq_cat(inputs)
    assert categoricals.shape == (batch_size, num_classifications, num_classes)

    quant_outputs = vq_codebook(categoricals)
    assert quant_outputs.shape == inputs.shape
