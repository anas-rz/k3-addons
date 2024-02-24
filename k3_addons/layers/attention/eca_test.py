import keras
from keras import ops
import pytest

from k3_addons.layers.attention.eca import ECAAttention


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 7, 7, 512), 
        (1, 14, 14, 256),
    ]
)
def test_eca(input_shape): 
    inputs = keras.random.uniform(input_shape) 
    eca = ECAAttention(kernel_size=3)
    output = eca(inputs)
    assert ops.shape(output) == input_shape 