from keras import layers, ops, Sequential, backend

from k3_addons.layers.pooling.adaptive_pooling import (
    AdaptiveMaxPool2D,
    AdaptiveAveragePool2D,
)
from k3_addons.api_export import k3_export

@k3_export('k3_addons.layers.ChannelAttention2D')
class ChannelAttention(layers.Layer):
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.maxpool = AdaptiveMaxPool2D(1)
        self.avgpool = AdaptiveAveragePool2D(1)
        self.data_format = backend.image_data_format()

    def build(self, input_shape):
        if self.data_format == "channels_last":
            input_dim = input_shape[3]
        else:
            input_dim = input_shape[1]
        self.se = Sequential(
            [
                layers.Conv2D(input_dim // self.reduction, 1, use_bias=False),
                layers.Activation("relu"),
                layers.Conv2D(input_dim, 1, use_bias=False),
            ]
        )

    def call(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = ops.sigmoid(max_out + avg_out)
        return output

@k3_export('k3_addons.layers.SpatialAttention2D')
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, padding='same')

    def call(self, x):
        if backend.image_data_format() == "channels_first":
            axis = 1
        else:
            axis = 3
        max_result = ops.max(x, axis=axis, keepdims=True)
        avg_result = ops.mean(x, axis=axis, keepdims=True)
        result = ops.concatenate([max_result, avg_result], axis)
        output = self.conv(result)
        output = ops.sigmoid(output)
        return output

@k3_export('k3_addons.layers.CBAM')
class CBAMBlock(layers.Layer):
    def __init__(self, reduction=16, kernel_size=49):
        super().__init__()
        self.channel_attention = ChannelAttention(reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        residual = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out + residual
