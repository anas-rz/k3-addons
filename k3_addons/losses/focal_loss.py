from keras import ops
from k3_addons.utils.keras_utils import LossFunctionWrapper

from k3_addons.api_export import k3_export


def sigmoid_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits: bool = False,
):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = ops.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = ops.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = ops.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = ops.cast(gamma, dtype=y_true.dtype)
        modulating_factor = ops.power((1.0 - p_t), gamma)

    # compute the final loss and return
    return ops.sum(alpha_factor * modulating_factor * ce, axis=-1)


@k3_export("k3_addons.losses.SigmoidFocalCrossEntropy")
class SigmoidFocalCrossEntropy(LossFunctionWrapper):
    def __init__(
        self,
        from_logits: bool = False,
        alpha=0.25,
        gamma=2.0,
        reduction=None,
        name="sigmoid_focal_crossentropy",
    ):
        super().__init__(
            sigmoid_focal_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma,
        )
