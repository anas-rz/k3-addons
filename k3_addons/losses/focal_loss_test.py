import pytest
import numpy as np
from keras import ops

from k3_addons.losses.focal_loss import SigmoidFocalCrossEntropy


@pytest.mark.parametrize(
    "y_true, y_pred",
    [([[1.0], [1.0], [0.0]], [[0.97], [0.91], [0.03]])],
)
def test_sigmoid_focal_crossentropy(y_true, y_pred):
    out_tf = ops.convert_to_tensor(
        [6.8532745e-06, 1.9097870e-04, 2.0559824e-05]
    )  # from tensorflow_addons
    # Calculate sigmoid within the test
    y_pred_sigmoid = y_pred

    # Use your focal loss implementation with the calculated sigmoid
    loss = SigmoidFocalCrossEntropy()(y_true=y_true, y_pred=y_pred_sigmoid)

    assert np.allclose(loss, out_tf)
