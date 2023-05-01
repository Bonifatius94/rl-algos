import numpy as np
from algos.dreamer.model import STGradsOneHotCategorical, ArgmaxLayer


def test_argmax():
    batch_size = 64
    num_classes = 10
    ohehot_categorical_input = np.zeros((batch_size, 32, num_classes))

    argmax = ArgmaxLayer()
    output = argmax(ohehot_categorical_input)

    assert output.shape == (batch_size, 32)


def test_onehot():
    batch_size = 64
    categorical_input = np.zeros((batch_size, 32, 16))

    sample = STGradsOneHotCategorical((32, 16))
    output = sample(categorical_input)

    assert output.shape == (batch_size, 32, 16)
    assert np.all(np.sum(output.numpy(), axis=-1) == np.ones((batch_size, 32)))
