from keras import layers, ops, Sequential
from k3_addons.layers.pooling.adaptive_pooling import AdaptiveAveragePool2D
from k3_addons.api_export import k3_export


@k3_export("k3_addons.layers.ECAAttention")
class ECAAttention(layers.Layer):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.pooling = AdaptiveAveragePool2D(1)
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, padding="same")

    def call(self, x):
        y = self.pooling(x)  # b,1, 1, c
        y = ops.squeeze(y, axis=2)  # b, 1, c
        y = ops.transpose(y, axes=[0, 2, 1])  # b, c, 1
        y = self.conv(y)  # b, c, 1
        y = ops.sigmoid(y)  # b, c, 1
        y = ops.transpose(y, axes=[0, 2, 1])  # b, 1, c
        y = ops.expand_dims(y, axis=2)  # bs, 1, 1, c
        return x * ops.broadcast_to(y, ops.shape(x))
