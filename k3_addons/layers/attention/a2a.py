from keras import layers, ops

from k3_addons.api_export import k3_export


@k3_export(path="k3_addons.layers.DoubleAttention")
class DoubleAttention(layers.Layer):
    """A2-Nets: Double Attention Networks [https://arxiv.org/pdf/1810.11579.pdf]"""

    def __init__(self, dim, value_dim=None, reconstruct=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.value_dim = value_dim or dim
        self.reconstruct = reconstruct

    def build(self, input_shape):
        self.in_dim = input_shape[-1]
        self.q = layers.Conv2D(self.dim, kernel_size=1)
        self.k = layers.Conv2D(self.value_dim, kernel_size=1)
        self.v = layers.Conv2D(self.value_dim, kernel_size=1)
        if self.reconstruct:
            self.conv_reconstruct = layers.Conv2D(self.in_dim, kernel_size=1)

    def call(self, x):
        b, h, w, c = ops.shape(x)
        assert c == self.in_dim, "input dim not equal to in_dim"
        q = self.q(x)  # b,h,w,dim
        q = ops.reshape(q, (b, h * w, self.dim))  # bnd
        k = self.k(x)  # b,h,w,vd
        k = ops.reshape(k, (b, h * w, self.value_dim))
        attention_maps = ops.softmax(k, axis=1)  # bnv
        v = self.v(x)  # b,h,w,vd
        v = ops.reshape(v, (b, h * w, self.value_dim))
        attention_vectors = ops.softmax(v, axis=1)  # b n vd
        global_descriptors = ops.einsum("bnd,bnv->bvd", q, attention_maps)  # b,d,vd
        out = ops.einsum("bvd,bnv->bnd", global_descriptors, attention_vectors)
        out = ops.reshape(out, (b, h, w, self.dim))  # b,h,w,c_n
        if self.reconstruct:
            out = self.conv_reconstruct(out)  # b,h,w,c
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "value_dim": self.value_dim,
                "reconstruct": self.reconstruct,
            }
        )
        return config
