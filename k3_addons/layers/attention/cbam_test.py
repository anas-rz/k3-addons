import pytest
import keras
from keras import ops


from k3_addons.layers.attention.cbam import ChannelAttention, SpatialAttention, CBAMBlock


@pytest.mark.parametrize("input_shape", [(1, 10, 10, 256), (1, 14, 14, 128)])
def test_channel_attention(input_shape):
    inputs = keras.random.normal(input_shape)  
    layer = ChannelAttention() 
    out = layer(inputs)
    assert ops.shape(out) == (1, 1, 1,) + (input_shape[-1],)  

@pytest.mark.parametrize("input_shape", [(1, 10, 10, 256), (1, 14, 14, 128)])
def test_spatial_attention(input_shape):
    inputs = keras.random.normal(input_shape)  
    layer = SpatialAttention() 
    out = layer(inputs)
    assert ops.shape(out) == input_shape[:-1] + (1,)  # Dynamic assertion

@pytest.mark.parametrize("input_shape", [(1, 10, 10, 256), (1, 14, 14, 128)])
def test_cbam(input_shape):
    inputs = keras.random.normal(input_shape)  # Modify input shape
    layer = CBAMBlock()
    out = layer(inputs)
    assert ops.shape(out) == input_shape  # Output shape should remain the same