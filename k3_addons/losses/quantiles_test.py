import numpy as np
from keras import ops
from k3_addons.losses.quantiles import PinballLoss


def test_pinball_loss():
    loss = PinballLoss(tau=.1)([0., 0., 1., 1.],[1., 1., 1., 0.])
    assert np.allclose(ops.convert_to_numpy(loss), 0.475)