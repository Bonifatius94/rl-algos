import os
import numpy as np

from algos.dreamer.config import DreamerSettings
from algos.dreamer.model import DreamerModel


def test_can_snapshot_and_restore_model():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)

    weights_dir = "temp_model"
    if os.path.exists(weights_dir):
        os.system(f"rm -rf {weights_dir}")

    test_state = np.zeros([batch_size] + settings.obs_dims)
    a_in = np.zeros([batch_size] + settings.action_dims)
    h_in = np.zeros([batch_size] + settings.hidden_dims)
    z_in = np.zeros([batch_size] + settings.repr_dims)
    inputs = (test_state, a_in, h_in, z_in)

    def assert_prediction_equal(model_1, model_2, inputs):
        model.seed(42)
        z1_hat_1, s1_hat_1, (r1_hat_1, term_hat_1), h1_1, z1_1 = model_1(inputs)
        model2.seed(42)
        z1_hat_2, s1_hat_2, (r1_hat_2, term_hat_2), h1_2, z1_2 = model_2(inputs)

        def arrays_equal(arr1, arr2):
            return np.all(arr1 == arr2)

        assert arrays_equal(z1_hat_1, z1_hat_2)
        assert arrays_equal(s1_hat_1, s1_hat_2)
        assert arrays_equal(r1_hat_1, r1_hat_2)
        assert arrays_equal(term_hat_1, term_hat_2)
        assert arrays_equal(h1_1, h1_2)
        assert arrays_equal(z1_1, z1_2)

    model.save(weights_dir)
    model2 = DreamerModel(settings)
    model2.load(weights_dir)

    assert os.path.exists(weights_dir)
    assert_prediction_equal(model.env_model, model2.env_model, inputs)

    os.system(f"rm -rf {weights_dir}")
