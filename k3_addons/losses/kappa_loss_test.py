from keras import ops
from k3_addons.losses.kappa_loss import WeightedKappaLoss

import numpy as np
def test_kappa_loss():
    y_true = ops.convert_to_tensor([[0, 0, 1, 0], [0, 1, 0, 0],   [1, 0, 0, 0], [0, 0, 0, 1]])
    y_pred = ops.convert_to_tensor([[0.1, 0.2, 0.6, 0.1], [0.1, 0.5, 0.3, 0.1],[0.8, 0.05, 0.05, 0.1], [0.01, 0.09, 0.1, 0.8]])
    kappa_loss = WeightedKappaLoss(num_classes=4)
    loss = kappa_loss(y_true, y_pred)
    np.allclose(loss.numpy(), -1.1611925)