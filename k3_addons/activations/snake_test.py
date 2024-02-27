import numpy as np
from k3_addons.activations.snake import snake
from keras import ops


def test_snake():
    x = ops.convert_to_tensor([-1.0, 0.0, 1.0])
    out = snake(x)
    assert np.allclose(ops.convert_to_numpy(out), [-0.29192656, 0.0, 1.7080734])
