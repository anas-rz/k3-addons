import pytest
import keras
from keras import ops
from k3_addons.layers.attention.parnet import ParNetAttention


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 128, 128, 3),
        (1, 64, 64, 16),
        (1, 32, 32, 32),
    ],
)
def test_parnet_attention(input_shape):
    inputs = keras.random.uniform(input_shape)
    layer = ParNetAttention()
    x = layer(inputs)

    assert ops.shape(x) == input_shape
