from keras import ops, layers
from k3_addons.api_export import k3_export


@k3_export(path="k3_addons.layers.ResidualAttention")
class ResidualAttention(layers.Layer):
    """Residual Attention: A Simple but Effective Method for Multi-Label Recognition [https://arxiv.org/abs/2108.02456]"""

    def __init__(self, num_class=1000, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.num_class = num_class
        self.fc = layers.Conv2D(
            self.num_class, kernel_size=1, strides=1, use_bias=False
        )

    def call(self, x):
        b, h, w, c = ops.shape(x)
        x = self.fc(x)
        x_raw = ops.reshape(x, (b, h * w, self.num_class))  # b, hxw, num_class

        x_avg = ops.mean(x_raw, axis=1)  # b,num_class
        x_max = ops.max(x_raw, axis=1)[0]  # b,num_class

        out = x_avg + self.alpha * x_max
        return out
