from keras import ops
from k3_addons.utils.keras_utils import LossFunctionWrapper
from k3_addons.api_export import k3_export


def contrastive_loss(y_true, y_pred, margin=1.0):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    return y_true * ops.square(y_pred) + (1.0 - y_true) * ops.square(
        ops.maximum(margin - y_pred, 0.0)
    )


@k3_export("k3_addons.losses.ContrastiveLoss")
class ContrastiveLoss(LossFunctionWrapper):
    def __init__(
        self,
        margin=1.0,
        reduction=None,
        name="contrastive_loss",
    ):
        super().__init__(
            contrastive_loss, reduction=reduction, name=name, margin=margin
        )
