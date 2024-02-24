import pytest
import keras
from k3_addons.layers.attention.a2a import DoubleAttention


@pytest.mark.parametrize(
    "dim, value_dim, reconstruct",
    [
        (128, 128, True),
        (128, 256, True),
        (256, 128, True),
        (256, 256, True),
    ],
)
def test_double_attn(dim, value_dim, reconstruct):
    input = keras.random.uniform(shape=(50, 7, 7, 512))
    a2 = DoubleAttention(dim, value_dim, reconstruct)
    output = a2(input)
    assert output.shape == input.shape
