import pytest
import keras
from keras import ops
from k3_addons.layers.attention.external_attention import ExternalAttention


def test_external_attention():
    inputs = keras.random.uniform(shape=(1, 50, 49, 512))
    ea = ExternalAttention(intermediate_dim=128)
    output = ea(inputs)

    # Explicit assertion for clarity and potential debugging
    assert output.shape == inputs.shape
