import pytest
import keras
from keras import ops
from k3_addons.layers.attention.simam import SimAM


@pytest.mark.parametrize(
    "shape",
    [
        (3, 7, 7, 64),
        (2, 14, 14, 32),
        (4, 5, 5, 128),
    ],
)
def test_sim_am_output_shape(shape):
    input_tensor = keras.random.normal(shape)
    layer = SimAM()
    outputs = layer(input_tensor)
    assert ops.shape(outputs) == ops.shape(input_tensor)
