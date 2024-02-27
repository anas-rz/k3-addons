from keras import ops
import numpy as np

from k3_addons.activations.lisht import lisht


def test_lisht():
    x = ops.convert_to_tensor([1.0, 0.0, 1.0])
    out = lisht(x)
    np.allclose(ops.convert_to_numpy(out), [0.7615942, 0.0, 0.7615942])
