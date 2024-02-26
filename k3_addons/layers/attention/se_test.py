import pytest
import keras
from keras import ops
from k3_addons.layers.attention.se import SEAttention


@pytest.mark.parametrize("reduction", [4, 8, 16])  # Test different reduction factors
def test_se_attention_output_shape(reduction):
    input_tensor = keras.random.normal((50, 7, 7, 512))
    se = SEAttention(reduction=reduction)
    output = se(input_tensor)
    assert ops.shape(output) == ops.shape(input_tensor)
