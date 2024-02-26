import keras
import pytest

from k3_addons.layers.normalization.instance_normalization import InstanceNormalization


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 196, 196),  
        (1, 256, 128),  
    ],
)
def test_instance_normalization(input_shape):
    layer = InstanceNormalization()
    inputs = keras.random.uniform(input_shape)  
    outputs = layer(inputs)
    assert outputs.shape == input_shape
