import numpy as np
from keras import ops
from k3_addons.activations.tanshrink import tanhshrink


def test_tanhshrink():
    x = ops.convert_to_tensor([-1.0, 0.0, 1.0])
    out = tanhshrink(x)
    assert np.allclose(ops.convert_to_numpy(out), [-0.23840582, 0.0, 0.23840582])
