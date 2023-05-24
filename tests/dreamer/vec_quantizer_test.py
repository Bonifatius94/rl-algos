import tensorflow as tf
from algos.dreamer.model import VectorQuantizer


def test_all_shapes_fine():
    inputs = tf.zeros((128, 8, 8, 64))

    vq = VectorQuantizer(16, 32)
    vq.build(inputs.shape)
    assert vq.embeddings.shape == (8 * 8 * 64 // 16, 16 * 32)

    outputs = vq(inputs)
    assert outputs.shape == inputs.shape
