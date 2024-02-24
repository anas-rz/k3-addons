import pytest
import keras
from k3_addons.layers.pooling.adaptive_pooling import (
    AdaptiveMaxPool1D,
    AdaptiveAveragePool1D,
    AdaptiveMaxPool2D,
    AdaptiveAveragePool2D,
)

from keras import ops

# Constants for clarity
CHANNELS_LAST = "channels_last"
CHANNELS_FIRST = "channels_first"


@pytest.mark.parametrize(
    "pool_layer, output_size, data_format, expected_shape",
    [
        (AdaptiveMaxPool1D, 5, CHANNELS_LAST, (1, 5, 1)),
        (AdaptiveAveragePool1D, 5, CHANNELS_LAST, (1, 5, 1)),
        (AdaptiveMaxPool1D, 5, CHANNELS_FIRST, (1, 1, 5)),
        (AdaptiveAveragePool1D, 5, CHANNELS_FIRST, (1, 1, 5)),
        (AdaptiveMaxPool1D, 1, CHANNELS_LAST, (1, 1, 1)),
        (AdaptiveAveragePool1D, 1, CHANNELS_FIRST, (1, 1, 1)),
    ],
)
def test_adaptive_pooling_1d(pool_layer, output_size, data_format, expected_shape):
    inputs = keras.random.uniform(shape=(1, 10, 1))
    if data_format == CHANNELS_FIRST:
        inputs = ops.transpose(inputs, (0, 2, 1))

    out = pool_layer(output_size=output_size, data_format=data_format)(inputs)
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "pool_layer, output_size, data_format, expected_shape",
    [
        (AdaptiveMaxPool2D, (5, 4), CHANNELS_LAST, (1, 5, 4, 3)),
        (AdaptiveAveragePool2D, (5, 4), CHANNELS_LAST, (1, 5, 4, 3)),
        (AdaptiveMaxPool2D, (5, 4), CHANNELS_FIRST, (1, 3, 5, 4)),
        (AdaptiveAveragePool2D, (5, 4), CHANNELS_FIRST, (1, 3, 5, 4)),
        (AdaptiveMaxPool2D, (1, 1), CHANNELS_LAST, (1, 1, 1, 3)),
        (AdaptiveAveragePool2D, (1, 1), CHANNELS_LAST, (1, 1, 1, 3)),
    ],
)
def test_adaptive_pooling_2d(pool_layer, output_size, data_format, expected_shape):
    inputs = keras.random.uniform(shape=(1, 10, 10, 3))
    if data_format == CHANNELS_FIRST:
        inputs = ops.transpose(inputs, (0, 3, 1, 2))

    out = pool_layer(output_size=output_size, data_format=data_format)(inputs)
    assert out.shape == expected_shape
