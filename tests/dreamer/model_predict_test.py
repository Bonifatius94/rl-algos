from typing import Iterable
import numpy as np

from algos.dreamer.config import DreamerSettings
from algos.dreamer.model import DreamerModelComponents


def init_inputs(batch_size: int, settings: DreamerSettings):
    test_state = np.zeros([batch_size] + settings.obs_dims)
    a_in = np.zeros([batch_size] + settings.action_dims)
    h_in = np.zeros([batch_size] + settings.hidden_dims)
    z_in = np.zeros([batch_size] + settings.repr_dims)
    return test_state, a_in, h_in, z_in


def init_model_comps(settings: DreamerSettings):
    model = DreamerModelComponents(settings)
    return model.compose_models()


def shape_of(array: np.ndarray, exp_shape: Iterable[int]) -> bool:
    return all([array.shape[i] == exp_shape[i] for i in range(len(array.shape))])


def test_prediction_with_env_model_has_expected_shapes():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    env_model, _, __, ___ = init_model_comps(settings)
    test_state, a_in, h_in, z_in = init_inputs(batch_size, settings)

    z1_hat, s1_hat, (r1_hat, term_hat), h1, z1 = env_model((test_state, a_in, h_in, z_in))

    assert shape_of(z1_hat.numpy(), [batch_size] + settings.repr_dims)
    assert shape_of(z1.numpy(), [batch_size] + settings.repr_dims)
    assert shape_of(s1_hat.numpy(), [batch_size] + settings.obs_dims)
    assert shape_of(h1.numpy(), [batch_size] + settings.hidden_dims)
    assert shape_of(r1_hat.numpy(), [batch_size, 1])
    assert shape_of(term_hat.numpy(), [batch_size, 1])


def test_prediction_with_dream_model_has_expected_shapes():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    _, dream_model, __, ___ = init_model_comps(settings)
    _, a_in, h_in, z_in = init_inputs(batch_size, settings)

    (r1_hat, term_hat), h1, z1_hat = dream_model((a_in, h_in, z_in))

    assert shape_of(z1_hat.numpy(), [batch_size] + settings.repr_dims)
    assert shape_of(h1.numpy(), [batch_size] + settings.hidden_dims)
    assert shape_of(r1_hat.numpy(), [batch_size, 1])
    assert shape_of(term_hat.numpy(), [batch_size, 1])


def test_prediction_with_step_model_has_expected_shapes():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    _, __, step_model, ___ = init_model_comps(settings)
    test_state, a_in, h_in, z_in = init_inputs(batch_size, settings)

    h1, z1 = step_model((test_state, a_in, h_in, z_in))

    assert shape_of(h1.numpy(), [batch_size] + settings.hidden_dims)
    assert shape_of(z1.numpy(), [batch_size] + settings.repr_dims)


def test_prediction_with_render_model_has_expected_shapes():
    batch_size = 32
    settings = DreamerSettings([10], [32, 32, 3], [32, 32], [512], [64])
    _, __, ___, render_model = init_model_comps(settings)
    _, __, h_in, z_in = init_inputs(batch_size, settings)

    s1_hat = render_model((h_in, z_in))

    assert shape_of(s1_hat.numpy(), [batch_size] + settings.obs_dims)
