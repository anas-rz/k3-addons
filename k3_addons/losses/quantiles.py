from keras import ops
from k3_addons.utils.keras_utils import LossFunctionWrapper
from k3_addons.api_export import k3_export


@k3_export("k3_addons.losses.pinball_loss")
def pinball_loss(y_true, y_pred, tau=0.5):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    tau = ops.expand_dims(ops.cast(tau, y_pred.dtype), 0)
    one = ops.cast(1, tau.dtype)

    delta_y = y_true - y_pred
    pinball = ops.maximum(tau * delta_y, (tau - one) * delta_y)
    return ops.mean(pinball, axis=-1)


@k3_export("k3_addons.losses.PinballLoss")
class PinballLoss(LossFunctionWrapper):
    def __init__(
        self,
        tau=0.5,
        reduction=None,
        name="pinball_loss",
    ):
        super().__init__(pinball_loss, reduction=reduction, name=name, tau=tau)
