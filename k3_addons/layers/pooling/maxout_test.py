import pytest
import keras
from keras import ops
from k3_addons.layers.pooling.maxout import Maxout


@pytest.mark.parametrize(
    "num_units, axis",
    [
        (8, -1),
        (4, -1),
        (16, -1),
        (8, -2),
    ],
)
def test_maxout_output_shape(num_units, axis):
    input_shape = (1, 224, 224, 32)
    inputs = keras.random.uniform(input_shape)
    out = Maxout(num_units, axis=axis)(inputs)

    # Construct the expected output shape
    expected_shape = list(input_shape)
    expected_shape[axis] = num_units

    assert ops.shape(out) == tuple(expected_shape)
