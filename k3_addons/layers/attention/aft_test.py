import pytest
import keras
from keras import ops

from k3_addons.layers.attention.aft import AFTFull


@pytest.mark.parametrize(
    "position_bias, projection_dim",
    [
        (True, 512),
        (False, 128),
    ],
)
def test_aft_full_position_bias(projection_dim, position_bias):
    projection_dim = 512
    inputs = keras.random.uniform((50, 49, projection_dim))

    layer = AFTFull(projection_dim=projection_dim, position_bias=position_bias)
    out = layer(inputs)

    assert ops.shape(out) == ops.shape(inputs)
