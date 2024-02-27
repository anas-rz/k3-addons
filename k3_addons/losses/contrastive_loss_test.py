import numpy as np
from keras import ops
from k3_addons.losses.contrastive_loss import ContrastiveLoss


def test_contrastive_loss():
    a = ops.convert_to_tensor([2, 3, 5], dtype="float16")
    b = ops.convert_to_tensor([5, 3, 1], dtype="float16")
    loss = ContrastiveLoss()(a, b)
    assert ops.shape(loss) == (3,)
    assert np.allclose(
        loss, ops.convert_to_tensor([50.0, 27.0, 5.0], dtype="float16")
    )  # from tf_addons output
