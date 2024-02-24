import keras
from keras import ops

from k3_addons.layers.attention.residual import ResidualAttention


def test_residual_attention():
    num_class = 1000
    alpha = 0.2  # Assign alpha for clarity
    inputs = keras.random.uniform((50, 7, 7, 512))

    layer = ResidualAttention(num_class=num_class, alpha=alpha)
    out = layer(inputs)

    assert out.shape == (50, num_class)
