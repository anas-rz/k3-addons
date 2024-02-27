import numpy as np
from keras import ops

from k3_addons.losses.giou_loss import GIoULoss


def test_sigmoid_giou():
    boxes1 = ops.convert_to_tensor([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = ops.convert_to_tensor([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])

    out_tfa = ops.convert_to_tensor([1.075, 1.9333334])

    # Use your focal loss implementation with the calculated sigmoid
    loss = GIoULoss()(boxes1, boxes2)

    assert np.allclose(loss, out_tfa)
