from keras import ops
from k3_addons.metrics.utils import MeanMetricWrapper
from k3_addons.api_export import k3_export


def hamming_distance(actuals, predictions):
    result = ops.not_equal(actuals, predictions)
    not_eq = ops.sum(ops.cast(result, "float32"))
    ham_distance = ops.divide_no_nan(not_eq, len(result))
    return ham_distance


def hamming_loss_fn(
    y_true,
    y_pred,
    threshold,
    mode,
):
    if mode not in ["multiclass", "multilabel"]:
        raise TypeError("mode must be either multiclass or multilabel]")

    if threshold is None:
        threshold = ops.max(y_pred, axis=-1, keepdims=True)
        # make sure [0, 0, 0] doesn't become [1, 1, 1]
        # Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred = ops.logical_and(y_pred >= threshold, ops.abs(y_pred) > 1e-12)
    else:
        y_pred = y_pred > threshold

    y_true = ops.cast(y_true, "int32")
    y_pred = ops.cast(y_pred, "int32")

    if mode == "multiclass":
        nonzero = ops.cast(ops.count_nonzero(y_true * y_pred, axis=-1), "float32")
        return 1.0 - nonzero

    else:
        nonzero = ops.cast(ops.count_nonzero(y_true - y_pred, axis=-1), "float32")
        return nonzero / ops.shape(y_true)[-1]


@k3_export("k3_addons.metrics.HammingLoss")
class HammingLoss(MeanMetricWrapper):
    def __init__(
        self,
        mode,
        name="hamming_loss",
        threshold=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            hamming_loss_fn, name=name, dtype=dtype, mode=mode, threshold=threshold
        )
