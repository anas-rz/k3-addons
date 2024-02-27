from keras import layers, ops, Sequential
from k3_addons.api_export import k3_export


def PreNorm(fn):
    def _apply(x):
        x = layers.LayerNormalization()(x)
        return fn(x)

    return _apply


def MLP(mlp_dim, dropout):
    def _apply(x):
        dim = ops.shape(x)[-1]
        net = Sequential(
            [
                layers.Dense(mlp_dim),
                layers.Activation("silu"),
                layers.Dropout(dropout),
                layers.Dense(dim),
                layers.Dropout(dropout),
            ]
        )
        return net(x)

    return _apply


class Attention(layers.Layer):
    def __init__(self, heads, head_dim, dropout):
        super().__init__()
        self.inner_dim = heads * head_dim
        self.head_dim = head_dim
        self.heads = heads
        self.scale = head_dim**-0.5
        self.dropout = dropout

    def build(self, input_shape):
        dim = input_shape[-1]
        self.to_qkv = layers.Dense(self.inner_dim * 3, use_bias=False)
        project_out = not (self.heads == 1 and self.head_dim == dim)
        self.to_out = (
            Sequential([layers.Dense(dim), layers.Dropout(self.dropout)])
            if project_out
            else layers.Identity()
        )

    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = ops.split(qkv, 3, axis=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = ops.transpose(
            ops.reshape(q, [-1, ops.shape(x)[1], self.heads, self.head_dim]),
            [0, 2, 1, 3],
        )
        k = ops.transpose(
            ops.reshape(k, [-1, ops.shape(x)[1], self.heads, self.head_dim]),
            [0, 2, 1, 3],
        )
        v = ops.transpose(
            ops.reshape(v, [-1, ops.shape(x)[1], self.heads, self.head_dim]),
            [0, 2, 1, 3],
        )
        dots = ops.matmul(q, ops.transpose(k, axes=(0, 1, 3, 2))) * self.scale
        attn = ops.softmax(dots)
        out = ops.matmul(attn, v)
        out = ops.reshape(out, [-1, x.shape[1], self.heads * self.head_dim])
        return self.to_out(out)


def Transformer(depth, heads, head_dim, mlp_dim, dropout=0.0):
    def _apply(x):
        x_alt = x
        for _ in range(depth):
            x_alt += PreNorm(Attention(heads, head_dim, dropout))(x_alt)
            x_alt += PreNorm(MLP(mlp_dim, dropout))(x_alt)
        return x_alt

    return _apply


@k3_export(path="k3_addons.layers.MobileViTAttention")
class MobileViTAttention(layers.Layer):
    def __init__(
        self,
        dim=512,
        kernel_size=3,
        patch_size=7,
        depth=3,
        heads=8,
        head_dim=64,
        mlp_dim=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ph, self.pw = patch_size, patch_size
        self.dim = dim
        self.kernel_size = kernel_size
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        in_channel = input_shape[-1]
        assert input_shape[1] % self.ph == 0
        self.conv1 = layers.Conv2D(
            in_channel, kernel_size=self.kernel_size, padding="same"
        )
        self.conv2 = layers.Conv2D(self.dim, kernel_size=1)

        self.trans = Transformer(
            depth=self.depth,
            heads=self.heads,
            head_dim=self.head_dim,
            mlp_dim=self.mlp_dim,
        )

        self.conv3 = layers.Conv2D(in_channel, kernel_size=1)
        self.conv4 = layers.Conv2D(
            in_channel, kernel_size=self.kernel_size, padding="same"
        )

    def call(self, x):
        x_skip = x
        x = self.conv2(self.conv1(x))

        b, h, w, c = ops.shape(x)
        x = ops.reshape(
            x, [b, h // self.ph, self.ph, w // self.pw, self.pw, ops.shape(x)[-1]]
        )
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [b, (h // self.ph) * (w // self.pw), -1])
        x = self.trans(x)
        x = ops.reshape(x, [b, h // self.ph, w // self.pw, self.ph, self.pw, -1])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [b, h, w, c])
        x = self.conv3(x)
        x = ops.concatenate([x, x_skip], axis=-1)
        x = self.conv4(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "ph": self.ph,
                "pw": self.pw,
                "dim": self.dim,
                "kernel_size": self.kernel_size,
                "patch_size": self.patch_size,
                "depth": self.depth,
                "heads": self.heads,
                "head_dim": self.head_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config
