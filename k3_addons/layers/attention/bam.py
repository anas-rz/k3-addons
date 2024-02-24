from keras import layers, ops, Sequential, backend
from k3_addons.layers.pooling.adaptive_pooling import AdaptiveAveragePool2D
from k3_addons.api_export import k3_export


class ChannelAttention(layers.Layer):
    def __init__(self, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = AdaptiveAveragePool2D(1)
        self.reduction = reduction
        self.num_layers = num_layers

    def build(self, input_shape):
        input_dim = input_shape[-1]
        gate_dims = [input_dim]
        gate_dims += [input_dim // self.reduction] * self.num_layers
        gate_dims += [input_dim]

        self.channel_attention = Sequential()
        self.channel_attention.add(layers.Flatten())
        for i in range(len(gate_dims) - 2):
            self.channel_attention.add(layers.Dense(gate_dims[i + 1]))
            self.channel_attention.add(layers.BatchNormalization())
            self.channel_attention.add(layers.Activation("relu"))
        self.channel_attention.add(layers.Dense(gate_dims[-1]))

    def call(self, x):
        if backend.image_data_format() == "channels_last":
            start_axis = 1
        else:
            start_axis = 2
        res = self.avgpool(x)  # b 1 1 c
        res = self.channel_attention(res)  # b c
        res = ops.expand_dims(res, axis=start_axis)  # b 1 c
        res = ops.expand_dims(res, axis=start_axis + 1)  # b 1 1 c
        res = ops.broadcast_to(res, ops.shape(x))
        return res


class SpatialAttention(layers.Layer):
    def __init__(self, reduction=16, num_layers=3, dilation_rate=2):
        super().__init__()

        self.reduction = reduction
        self.num_layers = num_layers
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        if backend.image_data_format() == "channels_last":
            input_dims = input_shape[-1]
        else:
            input_dims = input_shape[1]
        self.spatial_attention = Sequential()
        self.spatial_attention.add(
            layers.Conv2D(input_dims // self.reduction, kernel_size=1)
        )
        self.spatial_attention.add(layers.BatchNormalization())
        self.spatial_attention.add(layers.Activation("relu"))
        for i in range(self.num_layers):
            self.spatial_attention.add(layers.ZeroPadding2D(padding=1))
            self.spatial_attention.add(
                layers.Conv2D(
                    input_dims // self.reduction,
                    kernel_size=3,
                    dilation_rate=self.dilation_rate,
                )
            )
            self.spatial_attention.add(layers.BatchNormalization())
            self.spatial_attention.add(layers.Activation("relu"))
        self.spatial_attention.add(layers.Conv2D(1, kernel_size=1))

    def call(self, x):
        res = self.spatial_attention(x)
        res = ops.broadcast_to(res, ops.shape(x))
        return res


@k3_export(path="k3_addons.layers.BAMBlock")
class BAMBlock(layers.Layer):
    """
    BAM: Bottleneck Attention Module [https://arxiv.org/pdf/1807.06514.pdf]

    """

    def __init__(self, reduction=16, dilation_rate=2):
        super().__init__()
        self.channel_attention = ChannelAttention(reduction=reduction)
        self.spatial_attention = SpatialAttention(
            reduction=reduction, dilation_rate=dilation_rate
        )

    def call(self, x):
        sa_out = self.channel_attention(x)
        ca_out = self.spatial_attention(x)
        weight = ops.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out
