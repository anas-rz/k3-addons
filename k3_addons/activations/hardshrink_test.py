from keras import ops
import numpy as np
from k3_addons.activations.hardshrink import hardshrink


def test_hardshrink():
    x = ops.convert_to_tensor([1.0, 0.0, 1.0])
    out = hardshrink(x)
    assert np.allclose(ops.convert_to_numpy(out), [1.0, 0.0, 1.0])
