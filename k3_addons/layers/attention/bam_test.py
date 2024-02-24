import pytest
import keras
from keras import ops

from k3_addons.layers.attention.bam import BAMBlock


@pytest.mark.parametrize("input_shape", [(1, 7, 7, 512), (1, 7, 7, 128)])
def test_bam(input_shape):
    inputs = keras.random.uniform((input_shape))
    layer = BAMBlock(reduction=8)
    outputs = layer(inputs)
    assert ops.shape(outputs) == input_shape
