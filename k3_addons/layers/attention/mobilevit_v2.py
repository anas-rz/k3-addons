from keras import layers, ops
from k3_addons.api_export import k3_export


@k3_export(path="k3_addons.layers.MobileViTv2Attention")
class MobileViTv2Attention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        projection_dim = input_shape[-1]
        self.fc_i = layers.Dense(1)
        self.fc_k = layers.Dense(projection_dim)
        self.fc_v = layers.Dense(projection_dim)
        self.fc_o = layers.Dense(projection_dim)

    def call(self, input):
        i = self.fc_i(input)
        weight_i = ops.softmax(i, axis=1)
        context_score = weight_i * self.fc_k(input)
        context_vector = ops.sum(context_score, axis=1, keepdims=True)
        v = self.fc_v(input) * context_vector
        out = self.fc_o(v)

        return out
