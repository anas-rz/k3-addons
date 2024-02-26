from keras import layers, ops, Sequential
from k3_addons.layers.pooling.adaptive_pooling import AdaptiveAveragePool2D


class SEAttention(layers.Layer):
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.avg_pool = AdaptiveAveragePool2D(1)
        self.fc = Sequential(
            [
                layers.Dense(
                    input_dim // self.reduction, use_bias=False, activation="relu"
                ),
                layers.Dense(input_dim, use_bias=False, activation="sigmoid"),
            ]
        )

    def call(self, x):
        x_skip = x
        b, h, w, c = ops.shape(x)
        x = self.avg_pool(x)
        x = ops.reshape(x, (b, c))
        x = self.fc(x)
        x = ops.expand_dims(x, axis=1)
        x = ops.expand_dims(x, axis=2)
        return x_skip * ops.broadcast_to(x, ops.shape(x_skip))
