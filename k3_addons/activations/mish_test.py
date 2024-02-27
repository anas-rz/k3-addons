from keras import ops
import numpy as np
from k3_addons.activations.mish import mish


def test_mish():
    x = ops.convert_to_tensor([1.0, 0.0, 1.0])
    out = mish(x)
    assert np.allclose(ops.convert_to_numpy(out), [0.865098, 0.0, 0.865098])
