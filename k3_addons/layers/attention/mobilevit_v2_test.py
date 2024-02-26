import pytest
import keras
from keras import ops
from k3_addons.layers.attention.mobilevit_v2 import MobileViTv2Attention


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 100, 196),
        (2, 64, 128),
        (3, 32, 64),
    ],
)
def test_mobilevit_attention_output_shape(input_shape):
    inputs = keras.random.uniform(shape=input_shape)
    out = MobileViTv2Attention()(inputs)
    assert ops.shape(out) == ops.shape(inputs)
