from typing import Iterable
import numpy as np
from algos.dreamer import DreamerSettings, DreamerModel


def test_env_model_predict_shapes_correct():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    model = DreamerModel(settings)

    test_state = np.zeros([batch_size] + settings.obs_dims)
    a_in = np.zeros([batch_size] + settings.action_dims)
    h_in = np.zeros([batch_size] + settings.hidden_dims)
    z_in = np.zeros([batch_size] + settings.repr_dims)
    z1_hat, s1_hat, (r1_hat, term_hat), h1, z1 = model.env_model((test_state, a_in, h_in, z_in))

    def shape_of(array: np.ndarray, exp_shape: Iterable[int]) -> bool:
        return all([array.shape[i] == exp_shape[i] for i in range(len(array.shape))])

    assert shape_of(z1_hat.numpy(), [batch_size] + settings.repr_dims)
    assert shape_of(z1.numpy(), [batch_size] + settings.repr_dims)
    assert shape_of(s1_hat.numpy(), [batch_size] + settings.obs_dims)
    assert shape_of(h1.numpy(), [batch_size] + settings.hidden_dims)
    assert shape_of(r1_hat.numpy(), [batch_size, 1])
    assert shape_of(term_hat.numpy(), [batch_size, 1])


# def test_dream_model_predict_shapes_correct():
#     batch_size = 32
#     settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
#     model = DreamerModel(settings)

#     a_in = np.zeros([batch_size] + settings.action_dims)
#     h_in = np.zeros([batch_size] + settings.hidden_dims)
#     z_in = np.zeros([batch_size] + settings.repr_dims)
#     z1_hat, s1_hat, (r1_hat, term_hat), h1 = model.dream_model((a_in, h_in, z_in))

#     def shape_of(array: np.ndarray, exp_shape: Iterable[int]) -> bool:
#         return all([array.shape[i] == exp_shape[i] for i in range(len(array.shape))])

#     assert shape_of(z1_hat.numpy(), [batch_size] + settings.repr_dims)
#     assert shape_of(s1_hat.numpy(), [batch_size] + settings.obs_dims)
#     assert shape_of(h1.numpy(), [batch_size] + settings.hidden_dims)
#     assert shape_of(r1_hat.numpy(), [batch_size, 1])
#     assert shape_of(term_hat.numpy(), [batch_size, 1])
