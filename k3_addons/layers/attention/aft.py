from keras import layers, ops
from k3_addons.api_export import k3_export


@k3_export(path="k3_addons.layers.AFTFull")
class AFTFull(layers.Layer):
    """An Attention Free Transformer [https://arxiv.org/pdf/2105.14103v1.pdf]"""

    def __init__(self, projection_dim, position_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.position_bias = position_bias
        self.projection_dim = projection_dim
        self.to_q = layers.Dense(projection_dim)
        self.to_k = layers.Dense(projection_dim)
        self.to_v = layers.Dense(projection_dim)

    def build(self, input_shape):
        seq_len = input_shape[1]
        if self.position_bias:
            self.position_biases = self.add_weight(
                (1, seq_len, self.projection_dim), "zeros"
            )

    def call(self, inputs):
        b, n, d = ops.shape(inputs)
        q = self.to_q(inputs)  # bs,n,dim
        k = self.to_k(inputs)
        k = ops.expand_dims(k, 0)  # 1,bs,n,dim
        v = self.to_v(inputs)
        v = ops.expand_dims(v, 0)  # 1,bs,n,dim
        if self.position_bias:
            k = k + self.position_biases
            v = v + self.position_biases
        numerator = ops.sum(ops.exp(k) * v, axis=2)  # n,bs,dim
        denominator = ops.sum(ops.exp(k), axis=2)  # n,bs,dim
        out = numerator / denominator  # n,bs,dim
        out = ops.sigmoid(q) * (ops.transpose(out, (1, 0, 2)))  # bs,n,dim
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "position_bias": self.position_bias,
            }
        )
        return config
