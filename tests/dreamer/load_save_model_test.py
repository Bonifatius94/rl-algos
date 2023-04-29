import os
import numpy as np
import tensorflow as tf
from algos.dreamer import DreamerSettings, DreamerModel


def test_env_model_save_and_load_weights():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)

    weights_file = "weights.h5"
    if os.path.exists(weights_file):
        os.remove(weights_file)

    test_state = np.zeros([batch_size] + settings.obs_dims)
    a_in = np.zeros([batch_size] + settings.action_dims)
    h_in = np.zeros([batch_size] + settings.hidden_dims)
    z_in = np.zeros([batch_size] + settings.repr_dims)
    inputs = (test_state, a_in, h_in, z_in)

    def assert_prediction_equal(model_1, model_2, inputs):
        tf.random.set_seed(42)
        z1_hat_1, s1_hat_1, (r1_hat_1, term_hat_1), h1_1, z1_1 = model_1(inputs)
        tf.random.set_seed(42)
        z1_hat_2, s1_hat_2, (r1_hat_2, term_hat_2), h1_2, z1_2 = model_2(inputs)

        def arrays_equal(arr1, arr2):
            return np.all(arr1 == arr2)

        assert arrays_equal(z1_hat_1, z1_hat_2)
        assert arrays_equal(s1_hat_1, s1_hat_2)
        assert arrays_equal(r1_hat_1, r1_hat_2)
        assert arrays_equal(term_hat_1, term_hat_2)
        assert arrays_equal(h1_1, h1_2)
        assert arrays_equal(z1_1, z1_2)

    model.env_model.save_weights(weights_file)
    model2 = DreamerModel(settings)
    model2.env_model.load_weights(weights_file)

    assert os.path.exists(weights_file)
    assert_prediction_equal(model.env_model, model2.env_model, inputs)

    os.remove(weights_file)


def test_dream_model_save_and_load_weights():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)

    weights_file = "weights.h5"
    if os.path.exists(weights_file):
        os.remove(weights_file)

    a_in = np.zeros([batch_size] + settings.action_dims)
    h_in = np.zeros([batch_size] + settings.hidden_dims)
    z_in = np.zeros([batch_size] + settings.repr_dims)
    inputs = (a_in, h_in, z_in)

    def assert_prediction_equal(model_1, model_2, inputs):
        tf.random.set_seed(42)
        z1_hat_1, s1_hat_1, (r1_hat_1, term_hat_1), h1_1 = model_1(inputs)
        tf.random.set_seed(42)
        z1_hat_2, s1_hat_2, (r1_hat_2, term_hat_2), h1_2 = model_2(inputs)

        def arrays_equal(arr1, arr2):
            return np.all(arr1 == arr2)

        assert arrays_equal(z1_hat_1, z1_hat_2)
        assert arrays_equal(s1_hat_1, s1_hat_2)
        assert arrays_equal(r1_hat_1, r1_hat_2)
        assert arrays_equal(term_hat_1, term_hat_2)
        assert arrays_equal(h1_1, h1_2)

    model.dream_model.save_weights(weights_file)
    model2 = DreamerModel(settings)
    model2.dream_model.load_weights(weights_file)

    assert os.path.exists(weights_file)
    assert_prediction_equal(model.dream_model, model2.dream_model, inputs)

    os.remove(weights_file)
