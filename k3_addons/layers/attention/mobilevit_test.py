import pytest
import keras

from keras import ops

from k3_addons.layers.attention.mobilevit import MobileViTAttention


@pytest.mark.parametrize(
    "input_shape, dim, kernel_size, patch_size, depth, heads, head_dim, mlp_dim",
    [
        # Case 1: Your standard configuration
        ((1, 49, 49, 3), 512, 3, 7, 3, 8, 64, 1024),
        # Case 2: Smaller dim and mlp_dim
        ((1, 28, 28, 16), 256, 3, 7, 2, 4, 32, 512),
        # Case 3: Varying depth and patch size
        ((1, 64, 64, 8), 1024, 4, 8, 4, 16, 128, 2048),
    ],
)
def test_mobilevit_attention(
    input_shape, dim, kernel_size, patch_size, depth, heads, head_dim, mlp_dim
):
    inputs = keras.random.uniform(input_shape)

    attn_layer = MobileViTAttention(
        dim=dim,
        kernel_size=kernel_size,
        patch_size=patch_size,
        depth=depth,
        heads=heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
    )

    outputs = attn_layer(inputs)
    assert ops.shape(outputs) == ops.shape(inputs)
