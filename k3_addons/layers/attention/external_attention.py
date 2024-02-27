from keras import layers, ops
from k3_addons.api_export import k3_export


@k3_export("k3_addons.layers.ExternalAttention")
class ExternalAttention(layers.Layer):
    """
    Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks [https://arxiv.org/abs/2105.02358]
    """

    def __init__(self, intermediate_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.mk = layers.Dense(self.intermediate_dim, use_bias=False)
        self.mv = layers.Dense(in_dim, use_bias=False)
        self.softmax = layers.Softmax(axis=1)

    def call(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / ops.sum(attn, axis=2, keepdims=True)
        out = self.mv(attn)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config
